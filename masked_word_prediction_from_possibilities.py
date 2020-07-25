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
	   list of possible replacements, 
	   BERT loaded model, 
	   BERT tokenizer
Output: best fit word from list of possibilities to replace [MASK]
'''
def find_best_mask_replacement(masked_sentence, possibilities, model, tokenizer):
	possibility2loss = {}

	masked_tokenized = tokenizer.tokenize(masked_sentence)
	masked_input_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(masked_tokenized)])

	for possibility in possibilities:
		replaced_sent = masked_sentence.replace('[MASK]', possibility)
		tokenize_input = tokenizer.tokenize(replaced_sent)
		tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

		# calculating loss for ground truth and masked
		loss_score = model(masked_input_tensor, masked_lm_labels=tensor_input).item()
		possibility2loss[possibility] = loss_score

	sorted_possibility2loss = [(k, v) for k, v in sorted(possibility2loss.items(), key=lambda item: item[1])]
	# return possibility with least loss
	return sorted_possibility2loss[0][0]


if __name__ == '__main__':
	# Load the model once
	model, tokenizer = load_bert_model()

	# Examples
	# masked_sentence = 'What are the applications [MASK] KNN?'
	masked_sentence = 'Why does KNN suffer [MASK] the curse of dimensionality?'

	possibilities = ['of', 'in', 'from', 'on']

	# Get prediction from possibilities
	best_fit = find_best_mask_replacement(masked_sentence, possibilities, model, tokenizer)

	print(masked_sentence.replace('[MASK]', best_fit))

	# Output:
	# What are the applications of KNN?
	# Why does KNN suffer from the curse of dimensionality?