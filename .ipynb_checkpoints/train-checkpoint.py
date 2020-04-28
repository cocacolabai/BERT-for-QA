import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import time
import tensorflow as tf

output_dir="../bert_model/"
train_file="./data/train.json" 
dev_file="./data/dev.json" 

max_epoch = 3
batch_size = 4
lr = 1e-4
weight_decay = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert_pretrain_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_pretrain_name)
model = BertForQuestionAnswering.from_pretrained(bert_pretrain_name).to(device)
optim = AdamW(model.parameters(), lr)

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x80:
      return True
    return False

class QADataset(Dataset):
  def __init__(self, path: str, tokenizer: BertTokenizer) -> None:
    self.tokenizer = tokenizer
    self.data = []
    with open(path) as f:
      for article in json.load(f)['data']:
        for para in article['paragraphs']:
          context = para["context"]
          
          for qa in para['qas']:
            qa_id = qa['id']
            question = qa['question']
            
            #truncate
            cLen = 509 - len(question)
            if len(contexts[i])>cLen:      
                context=context[:cLen]
            
            answerable = qa['answerable']
            if answerable:
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length -1]
                actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
                tf.compat.v1.logging.warning("Find answer: '%s' vs. '%s'",actual_text, orig_answer_text)
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""
            
            self.data.append((qa_id, context, question, orig_answer_text, answerable))
  
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question, answer, answerable = self.data[index]
    return qa_id, context, question, answer, int(answerable)
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def findInList(mylist, pattern):
    for i in range(len(mylist)):
      if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
        return i, i+len(pattern)
    return 0,0
    
train_dataset = QADataset(train_file, tokenizer)
valid_dataset = QADataset(dev_file, tokenizer)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)


# dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
# eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


best_valid_loss = float('inf')



for epoch in trange(max_epoch):

  start_time = time.time()
  model.train()
  pbar= tqdm(train_loader)
  for batch in pbar:
    ids, contexts, questions, answer, answerable = batch
    #print(ids)
    input_dict = tokenizer.batch_encode_plus([contexts, questions], 
                                             max_length=tokenizer.max_len, 
                                             pad_to_max_length=True,
                                             return_tensors='pt')
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
#     print(start_position, end_position)
    start_list=[]
    end_list=[]
    for i in range(len(ids)):
        text_encode = tokenizer.encode(answer[i])[1:-1]
        if text_encode!=[]:  
          start, end = findInList(input_dict["input_ids"][i].tolist(), text_encode)
          start_list.append(start)
          end_list.append(end)
        else:
          start_list.append(0)
          end_list.append(0)
            
    outputs = model(start_positions=torch.tensor(start_list).to(device), end_positions=torch.tensor(end_list).to(device),
                                 **input_dict)
    loss = outputs[0]
    loss.backward()
    optim.step()
    optim.zero_grad()
    pbar.set_description(f"train loss: {loss.item():.4f}")
  
  model.eval()
  with torch.no_grad():
    pbar=tqdm(valid_loader)
    for batch in pbar:
        ids, contexts, questions, answer, answerable = batch
        #print(ids)
        input_dict = tokenizer.batch_encode_plus([contexts, questions], 
                                                 max_length=tokenizer.max_len, 
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
    #     print(start_position, end_position)
        start_list=[]
        end_list=[]
        for i in range(len(ids)):
            text_encode = tokenizer.encode(answer[i])[1:-1]
            if text_encode!=[]:  
              start, end = findInList(input_dict["input_ids"][i].tolist(), text_encode)
              start_list.append(start)
              end_list.append(end)
            else:
              start_list.append(0)
              end_list.append(0)

        outputs = model(start_positions=torch.tensor(start_list).to(device), end_positions=torch.tensor(end_list).to(device),
                                     **input_dict)
        loss = outputs[0]
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


