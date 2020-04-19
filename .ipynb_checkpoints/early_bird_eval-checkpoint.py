import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()



max_epoch = 3
batch_size = 4
lr = 1e-4
weight_decay = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert_pretrain_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_pretrain_name)
model = BertForNextSentencePrediction.from_pretrained(bert_pretrain_name).to(device)

##### load checkpoint


optim = AdamW(model.parameters(), lr)

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

for epoch in trange(max_epoch):
    start_time = time.time()
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
        for i,id in zip(a[0]>0, ids):
            if i:
                all_predictions[id] = "have answer"
            else:
                all_predictions[id] = ""
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if loss < best_valid_loss:
        best_valid_loss = loss
        torch.save(model.state_dict(), '../early_model.pt')