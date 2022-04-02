import sys
sys.path.append("/home/code/")
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
from sklearn.preprocessing import MultiLabelBinarizer


from torch import nn
SEED_VALUE = 0

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def tokenize_function(text_examples, tokenizer, max_length):
    if isinstance(text_examples, pd.Series):
        text_examples = text_examples.tolist()

    return tokenizer(text_examples, truncation=True,
                          padding="max_length", max_length=max_length)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_vector, tokenizer, max_length):
        self.encodings = tokenize_function(df['text'], tokenizer, max_length)
        self.label_vector = label_vector
        self.df_len = df.shape[0]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label_vector'] = torch.tensor(self.label_vector[idx])
        return item

    def __len__(self):
        return self.df_len


class SimpleCls:
    def __init__(self, model_name, df_data, max_length, device):
        reset_seed(SEED_VALUE)

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.device = device
        self.encoder = AutoModel.from_pretrained(model_name)

        # convert labels to multi-hot vectors
        multi_hot_encoder = MultiLabelBinarizer()
        multi_hot_labels = multi_hot_encoder.fit_transform(df_data['postures'])

        # count occurrence of each label - for reference
        self.label_count = multi_hot_labels.sum(0)
        self.multi_hot_encoder = multi_hot_encoder
        self.num_labels = len(multi_hot_encoder.classes_)
        # initialize classification model
        self.model = ClassificationHead(self.encoder, self.num_labels, device)
        # split data to train and test
        x_train, x_test, y_train, y_test = train_test_split(df_data, multi_hot_labels, test_size=0.2,
                                                            random_state=SEED_VALUE)
        # resplit train data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                            random_state=SEED_VALUE)

        # Define a custom dataset for each data split
        self.train_data = CustomDataset(x_train, y_train, self.tokenizer, max_length=self.max_length)
        self.val_data = CustomDataset(x_val, y_val, self.tokenizer, max_length=self.max_length)
        self.test_data = CustomDataset(x_test, y_test, self.tokenizer, max_length=self.max_length)




    def train(self, batch_size:int=32, epochs:int=5):
        device = self.device
        self.model = ClassificationHead(self.encoder, self.num_labels, device)
        self.model.train()
        self.train_batch_size = batch_size
        self.train_epochs = epochs

        train_loader = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True)
        # val data loaded in batches to lower memory usage
        val_loader = DataLoader(self.val_data, batch_size=self.train_batch_size)

        # define optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=0.001)
        # define loss function.
        # Because this is a multi-label classification task, we will use a sigmoid based loss function as it treats each label independently
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        # simple training loop with validation after each epoch
        for epoch in range(self.train_epochs):
            train_loss = []
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.model.zero_grad()
                    tepoch.set_description(f"Epoch {epoch}")
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label_vector']
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs, labels.type(torch.FloatTensor).to(self.model.new_device))
                    loss.backward()
                    optim.step()
                    input_ids.detach()
                    attention_mask.detach()
                    train_loss.append(loss.item())
                    tepoch.set_postfix({'train_loss': np.mean(train_loss)}, refresh=True)

                with torch.no_grad():
                    # validate
                    predictions = []
                    val_labels = []
                    for batch in val_loader:
                        with torch.no_grad():
                            # for batch in data:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            batch_outputs = self.model(input_ids, attention_mask=attention_mask)
                            batch_outputs = batch_outputs.cpu().detach().tolist()
                            batch_labels = batch['label_vector'].tolist()
                            predictions += batch_outputs
                            val_labels += batch_labels
                    predictions = torch.tensor(predictions)
                    val_labels = torch.tensor(val_labels)
                    val_loss = loss_fn(predictions.to(device),
                                   val_labels.type(torch.FloatTensor).to(device))
                    print('Epoch {}, val_loss: {}'.format(epoch, val_loss.item()))
        self.model.eval()


    def test(self, threshold=0.5):
        test_loader = iter(DataLoader(self.test_data, batch_size=32))
        predictions = []
        labels = []
        for batch in test_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_outputs = batch_outputs.sigmoid().cpu().detach().tolist()
                batch_labels = batch['label_vector'].tolist()
                predictions += batch_outputs
                labels += batch_labels
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)
        # convert predictions with probability above >0.5 to 1, and with probability <0.5 to 0
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        print(classification_report(predictions, labels, target_names=self.multi_hot_encoder.classes_))



class ClassificationHead(torch.nn.Module):

    def __init__(self, encoder, n_classes, device=None):
        super(ClassificationHead, self).__init__()
        if device is None:
            self.new_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.new_device = device

        self.encoder = encoder
        try:
            emb_dim = self.encoder.config.dim
        except AttributeError:
            emb_dim = self.encoder.config.hidden_size

        if emb_dim < 1000:
            # if dimensions is <1000, then patch pool_distil
            self.pooler = ClassificationHead.pool_distil
        else:
            self.pooler = ClassificationHead.pool_bertlarge

        # Defining layers and activations
        self.layer0 = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.drop0 = torch.nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=emb_dim, out_features=n_classes)
        self.relu_act = nn.ReLU()
        self.to(self.new_device)


    @staticmethod
    def pool_distil(outputs):
        last_hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = last_hidden_state[:, 0]
        return pooled_output

    @staticmethod
    def pool_bertlarge(outputs):
        pooled_output = outputs[1]
        return pooled_output

    def forward(self,
        input_ids=None,
        attention_mask=None,
    ):
        attention_mask.to(self.new_device)
        input_ids.to(self.new_device)
        # pass input ids and attention mask to the transformer-based model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # get the output of the [CLS] token. This token should capture the input representation
        pooled_output = self.pooler(outputs)
        # apply the remaining steps
        x = self.layer0(pooled_output)
        x = self.relu_act(x)
        x = self.drop0(x)

        output = self.out(x)
        return output
