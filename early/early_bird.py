import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import time
import os

output_dir = '../model/'
max_epoch = 3
batch_size = 4
lr = 1e-4
weight_decay = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert_pretrain_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_pretrain_name)
model = BertForNextSentencePrediction.from_pretrained(bert_pretrain_name).to(device)
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
    
train_dataset = EarlyDataset("./data/train-small.json", tokenizer)
valid_dataset = EarlyDataset("./data/dev-small.json", tokenizer)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

best_valid_loss = float('inf')


for epoch in trange(max_epoch):

  start_time = time.time()
  model.train()
  pbar= tqdm(train_loader)
  for batch in pbar:
    ids, contexts, questions, answerable = batch
    input_dict = tokenizer.batch_encode_plus(contexts, questions, 
                                             max_length=tokenizer.max_len, 
                                             pad_to_max_length=True,
                                             return_tensors='pt')
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
    loss, logits = model(next_sentence_label=answerable.to(device), 
                         **input_dict)
#     print("logits:", logits[0])
    loss.backward()
    optim.step()
    optim.zero_grad()
    pbar.set_description(f"train loss: {loss.item():.4f}")
  
  model.eval()
  with torch.no_grad():
    pbar=tqdm(valid_loader)
    for batch in pbar:
        ids, contexts, questions, answerable = batch
        input_dict = tokenizer.batch_encode_plus(contexts, questions, 
                                                 max_length=tokenizer.max_len, 
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        loss, logits = model(next_sentence_label=answerable.to(device), 
                             **input_dict)

        pbar.set_description(f"val loss: {loss.item():.4f}")
    
  
  end_time = time.time()
  print("val loss:",loss)

  if loss < best_valid_loss:
        best_valid_loss = loss
        # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

        # If we have a distributed model, save only the encapsulated model
        # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`

        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

