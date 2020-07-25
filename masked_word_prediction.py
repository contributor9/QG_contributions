import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pdb

def load_bert_model():
	with torch.no_grad():
		model = BertForMaskedLM.from_pretrained('bert-base-uncased')
		model.eval()
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	return model, tokenizer


'''
Input: question with token replacd with [MASK] token for word to be predicted, 
	   BERT loaded model, 
	   BERT tokenizer
Output: word that can replace [MASK] token in question
'''
def get_masked_word(question, model, tokenizer):
	question = '[CLS] ' + question + ' [SEP]'

	tokenized_text = tokenizer.tokenize(question)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	masked_index = tokenized_text.index('[MASK]')

	# Create the segments tensors.
	segments_ids = [0] * len(tokenized_text)

	# Convert inputs to PyTorch tensors
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	# Predict all tokens
	with torch.no_grad():
		predictions = model(tokens_tensor, segments_tensors)

	predicted_index = torch.argmax(predictions[0, masked_index]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	return predicted_token


if __name__ == "__main__":
	# Load the model once
	model, tokenizer = load_bert_model()

	# Example questions
	# question = 'What are the applications [MASK] KNN?'
	# question = '[MASK] are the applications of KNN?'
	# question = 'Why does KNN suffer [MASK] the curse of dimensionality?'
	question = '[MASK] does KNN suffer from the curse of dimensionality?'

	# Get prediction for masked word
	word = get_masked_word(question, model, tokenizer)

	print(question.replace('[MASK]', word))

	# Output:
	# What are the applications of KNN?
	# what are the applications of KNN?
	# Why does KNN suffer from the curse of dimensionality?
	# why does KNN suffer from the curse of dimensionality?