import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import time
import tensorflow as tf
import math
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()


output_dir="../bert_model/"

batch_size = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

max_answer_length = 30
n_best_size = 20

tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForQuestionAnswering.from_pretrained(output_dir).to(device)



def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x80:
      return True
    return False

class QATestDataset(Dataset):
  def __init__(self, path: str, tokenizer: BertTokenizer) -> None:
    self.tokenizer = tokenizer
    self.data = []
    with open(args.test_data_path) as f:
      for article in json.load(f)['data']:
        for para in article['paragraphs']:
          context = para["context"]
          doc_tokens = []
          char_to_word_offset = []
          for c in context:
                doc_tokens.append(c)  # 每个token
                char_to_word_offset.append(len(doc_tokens) - 1)  # 每个字的index
    
#           print(char_to_word_offset, len(char_to_word_offset))
          for qa in para['qas']:
            qa_id = qa['id']
            question = qa['question']
            self.data.append((qa_id, context, question, doc_tokens, char_to_word_offset))
  
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question, doc_tokens, char_to_word_offset = self.data[index]
    return qa_id, context, question, doc_tokens, char_to_word_offset
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x, reverse=True)
  
  best_indexes = []
  for i in range(len(index_and_score)):
    #print(i, index_and_score[i][0])
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes

def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
#   print("scores:",scores)
  if not scores:
#     print("out")
    return []

  max_score = None
  for score in scores[0]:
#     print("score:",score)
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores[0]:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def get_final_text(pred_text, orig_text, do_lower_case=True):
  """Project the tokenized prediction back to the original text."""
  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = {}
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize orig_text, strip whitespace from the result
  # and pred_text, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    return orig_text

  # We then project the characters in pred_text back to orig_text using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in tok_ns_to_s_map.items():
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


  
test_dataset = QATestDataset(args.test_data_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



best_valid_loss = float('inf')

all_predictions = {}
all_nbest_json = {}


model.eval()
with torch.no_grad():
    output = {}
    pbar=tqdm(test_loader)
    for batch in pbar:
        ids, contexts, questions, doc_tokens, char_to_word_offset = batch
#         print("size:",len(char_to_word_offset),len(doc_tokens)) ####
        input_dict = tokenizer.batch_encode_plus(contexts, questions, 
                                                 max_length=tokenizer.max_len, 
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        logits = model(**input_dict)
        
        ###################
        score_null = 1000000  # large and positive ###
        prelim_predictions = []
        
#         print(logits[0].argmax(-1))
        start_index = logits[0].argmax(-1)
        end_index = logits[1].argmax(-1)

#         print("------------")
#         print(start_index,end_index)
            
        for i in range(len(ids)):
            if (start_index[i] < tokenizer.max_len) and (end_index[i] < tokenizer.max_len) and (end_index[i] > start_index[i]) and (start_index[i] > 0) and end_index[i] < len(char_to_word_offset):
                new_doc=[d[i] for d in doc_tokens]
                new_char=[c[i] for c in char_to_word_offset]
                #print("new:", new_doc, new_char)
                prelim_predictions.append((ids[i],start_index[i], end_index[i], logits[0][i], logits[1][i], new_doc, new_char))
                
        
#         for i in range(batch_size):
            
#             start_indexes = logits[0][i].argsort()[-n_best_size:][::-1]
#             end_indexes = logits[1][i].argsort()[-n_best_size:][::-1]
#       
#      print(start_indexes)
#             print(end_indexes)
            
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     if not (start_index >= tokenizer.max_len) and not (end_index >= tokenizer.max_len) and not (end_index < start_index) and (start_index > 0):
#                         prelim_predictions.append((start_index, end_index, logits[0][i], logits[1][i]))
        seen_predictions = {}
        
        nbest = []
        for pred in prelim_predictions:
          qa_id, start_index, end_index, start_logit, end_logit, doc_tokens, char_to_word_offset= pred
          #print("pred:",pred)
          if len(nbest) >= n_best_size:
            break
          if start_index > 0:  # this is a non-null prediction
            #print(len(char_to_word_offset),len(start_logit),tokenizer.max_len)
            tok_tokens = doc_tokens[start_index:(end_index + 1)]
            #print(tok_tokens)
            orig_doc_start = char_to_word_offset[start_index]
            orig_doc_end = char_to_word_offset[end_index]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = "".join(tok_tokens)
            print("tok_text:",tok_text)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = "".join(tok_text.split())
            orig_text = "".join(orig_tokens)
            final_text = get_final_text(tok_text, orig_text)
            if final_text in seen_predictions:
              continue
            seen_predictions[final_text] = True
          else:
            final_text = ""
            seen_predictions[final_text] = True

          nbest.append((qa_id,final_text,start_logit,end_logit))


        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
          qa_id,text,start_logit,end_logit = entry
          total_scores.append(start_logit + end_logit)
          if not best_non_null_entry:
            if text:
              best_non_null_entry = entry
#         print(total_scores,(total_scores[0]).size())
        probs = _compute_softmax(total_scores)
        print("probs:",probs)

        for (i, entry) in enumerate(nbest):
          qa_id,text,start_logit,end_logit = entry
          
          if probs[i]>0.3:
            output[qa_id] = text
          else:
            output[qa_id] = ""
        print(output)
    Path(args.output_path).write_text(json.dumps(output))


