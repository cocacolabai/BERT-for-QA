import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import time
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()

output_dir = '../model/'

batch_size = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForNextSentencePrediction.from_pretrained(output_dir).to(device)

class EarlyDataset(Dataset):
  def __init__(self, path: str, tokenizer: BertTokenizer) -> None:
    self.tokenizer = tokenizer
    self.data = []
    with open(path) as f:
      for article in json.load(f)['data']:
        parapraphs = article['paragraphs']
        for para in parapraphs:
          context = para['context']
          for qa in para['qas']:
            qa_id = qa['id']
            question = qa['question']
            answerable = qa['answerable']
            self.data.append((qa_id, context, question, answerable))
  
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question, answerable = self.data[index]
    return qa_id, context, question, int(answerable)
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
test_dataset = EarlyDataset(args.test_data_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

best_valid_loss = float('inf')

with torch.no_grad():
    pbar=tqdm(test_loader)
    for batch in pbar:
        ids, contexts, questions, answerable = batch
        input_dict = tokenizer.batch_encode_plus(contexts, questions, 
                                                 max_length=tokenizer.max_len, 
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        loss, logits = model(next_sentence_label=answerable.to(device), 
                             **input_dict)
        all_predictions = {}
        for i,id in zip(logits[0]>0, ids):
            if i:
                all_predictions[id] = "have answer"
            else:
                all_predictions[id] = ""

output_file=Path(args.output_path)
output_file.write_text(json.dumps(all_predictions))