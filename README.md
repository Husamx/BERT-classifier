# BERT-classifier

BERT Classifier.py contains an implementation of a BERT classifier for a multi-label classification task, where samples can be mapped to more than one target label.
## Example of usage
    model = Classifier.SimpleCls(model_name= ‘bert-base-uncased’, df_data=data, max_length=512, device='cuda:0')
bert-base-uncased is a 12-layer pretrained BERT model from Huggingface. Refer to https://huggingface.co/transformers/v4.1.1/pretrained_models.html <br/>
df_data is a pandas dataframe object. It should contain two columns: <br/>
text: Text content of data samples<br/>
label_vector: for each sample, this is a list with the target labels <br/>
max_length: the maximum number of input tokens to process per sample. This controls the length of the input sequence that the model considers for classification. 

## train
    model.train(epochs=10)

## eval
    model.test(threshold=0.5)
Output layer activated with a sigmoid function for multi-label classification. Hence, the sum of probabilities don’t add to 1. 
The threshold argument sets the decision boundary which is 0.5 in this example. 
if label is mapped to probability > 0.5 , the prediction is 1
if label is mapped to probability <= 0.5 , the prediction in 0

