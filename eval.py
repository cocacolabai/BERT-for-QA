import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import time
import tensorflow as tf
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()


output_dir="../bert_model/"

batch_size = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForNextSentencePrediction.from_pretrained(output_dir).to(device)



def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x80:
      return True
    return False

class QATestDataset(Dataset):
  def __init__(self, path: str, tokenizer: BertTokenizer) -> None:
    self.tokenizer = tokenizer
    self.data = []
    with open(path) as f:
      for article in json.load(f)['data']:
        parapraphs = article['paragraphs']
        
        ###########################
        for para in parapraphs:
          context = para['context']

          doc_tokens = []
          char_to_word_offset = []
          prev_is_whitespace = True
          for c in context:
            if is_whitespace(c):
              prev_is_whitespace = True
            else:
              if prev_is_whitespace:
                doc_tokens.append(c)
              else:
                doc_tokens[-1] += c
              prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        ###########################
        
          for qa in para['qas']:
            qa_id = qa['id']
            question = qa['question']
            self.data.append((qa_id, context, question, doc_tokens))
  
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question, doc_tokens = self.data[index]
    return qa_id, context, question, doc_tokens
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
test_dataset = QADataset(args.test_data_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



best_valid_loss = float('inf')

all_predictions = {}

model.eval()
with torch.no_grad():
    pbar=tqdm(test_loader)
    for batch in pbar:
        ids, contexts, questions, doc_tokens = batch
        input_dict = tokenizer.batch_encode_plus(contexts, questions, 
                                                 max_length=tokenizer.max_len, 
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        logits = model(**input_dict)[0]
        print(logits)
        pbar.set_description(f"val loss: {loss.item():.4f}")

# class SquadExample(object):
#   """A single training/test example for simple sequence classification.

#      For examples without an answer, the start and end position are -1.
#   """

#   def __init__(self,
#                qas_id,
#                question_text,
#                doc_tokens,
#                orig_answer_text=None,
#                start_position=None,
#                end_position=None,
#                answerable=True):
#     self.qas_id = qas_id
#     self.question_text = question_text
#     self.doc_tokens = doc_tokens
#     self.orig_answer_text = orig_answer_text
#     self.start_position = start_position
#     self.end_position = end_position
#     self.answerable = answerable

#   def __str__(self):
#     return self.__repr__()

#   def __repr__(self):
#     s = ""
#     s += "qas_id: %s" % (printable_text(self.qas_id))
#     s += ", question_text: %s" % (
#         printable_text(self.question_text))
#     s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
#     if self.start_position:
#       s += ", start_position: %d" % (self.start_position)
#     if self.start_position:
#       s += ", end_position: %d" % (self.end_position)
#     if self.start_position:
#       s += ", answerable: %r" % (self.answerable)
#     return s


# def whitespace_tokenize(text):
#   """Runs basic whitespace cleaning and splitting on a piece of text."""
#   text = text.strip()
#   if not text:
#     return []
#   tokens = text.split()
#   return tokens

# def printable_text(text):
#   """Returns text encoded in a way suitable for print or `tf.logging`."""

#   # These functions want `str` for both Python2 and Python3, but in one case
#   # it's a Unicode string and in the other it's a byte string.
#   if six.PY3:
#     if isinstance(text, str):
#       return text
#     elif isinstance(text, bytes):
#       return text.decode("utf-8", "ignore")
#     else:
#       raise ValueError("Unsupported string type: %s" % (type(text)))
#   elif six.PY2:
#     if isinstance(text, str):
#       return text
#     elif isinstance(text, unicode):
#       return text.encode("utf-8")
#     else:
#       raise ValueError("Unsupported string type: %s" % (type(text)))
#   else:
#     raise ValueError("Not running on Python2 or Python 3?")
    
    
# def read_squad_examples(input_file, is_training):
#   """Read a SQuAD json file into a list of SquadExample."""
#   with tf.compat.v1.gfile.Open(input_file, "r") as reader:
#     input_data = json.load(reader)["data"]

#   def is_whitespace(c):
#     if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x80:
#       return True
#     return False

#   examples = []

#   for entry in input_data:
#     for paragraph in entry["paragraphs"]:
#       paragraph_text = paragraph["context"]
#       doc_tokens = []
#       char_to_word_offset = []
#       prev_is_whitespace = True
#       for c in paragraph_text:
#         if is_whitespace(c):
#           prev_is_whitespace = True
#         else:
#           if prev_is_whitespace:
#             doc_tokens.append(c)
#           else:
#             doc_tokens[-1] += c
#           prev_is_whitespace = False
#         char_to_word_offset.append(len(doc_tokens) - 1)

#       for qa in paragraph["qas"]:
#         qas_id = qa["id"]
#         question_text = qa["question"]
#         start_position = None
#         end_position = None
#         orig_answer_text = None
#         answerable = True
#         if is_training:

#           if version_2_with_negative:
#             answerable = qa["answerable"]
# #           if (len(qa["answers"]) != 1) and answerable:
# #             raise ValueError(
# #                 "For training, each question should have exactly 1 answer.")
#           if answerable:
#             answer = qa["answers"][0]
#             orig_answer_text = answer["text"]
#             answer_offset = answer["answer_start"]
#             answer_length = len(orig_answer_text)
#             start_position = char_to_word_offset[answer_offset]
#             end_position = char_to_word_offset[answer_offset + answer_length -
#                                                1]
#             # Only add answers where the text can be exactly recovered from the
#             # document. If this CAN'T happen it's likely due to weird Unicode
#             # stuff so we will just skip the example.
#             #
#             # Note that this means for training mode, every example is NOT
#             # guaranteed to be preserved.
#             actual_text = " ".join(
#                 doc_tokens[start_position:(end_position + 1)])
#             cleaned_answer_text = " ".join(
#                 whitespace_tokenize(orig_answer_text))
#             if actual_text.find(cleaned_answer_text) == -1:
#               tf.compat.v1.logging.warning("Could not find answer: '%s' vs. '%s'",
#                                  actual_text, cleaned_answer_text)
#               continue
#           else:
#             start_position = -1
#             end_position = -1
#             orig_answer_text = ""

#         example = SquadExample(
#             qas_id=qas_id,
#             question_text=question_text,
#             doc_tokens=doc_tokens,
#             orig_answer_text=orig_answer_text,
#             start_position=start_position,
#             end_position=end_position,
#             answerable=answerable)
#         examples.append(example)
#   return examples


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.compat.v1.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.compat.v1.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if  version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    if  version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if  version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not  version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff >  null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if  version_2_with_negative:
    with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


# def get_final_text(pred_text, orig_text, do_lower_case):
#   """Project the tokenized prediction back to the original text."""

#   # When we created the data, we kept track of the alignment between original
#   # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
#   # now `orig_text` contains the span of our original text corresponding to the
#   # span that we predicted.
#   #
#   # However, `orig_text` may contain extra characters that we don't want in
#   # our prediction.
#   #
#   # For example, let's say:
#   #   pred_text = steve smith
#   #   orig_text = Steve Smith's
#   #
#   # We don't want to return `orig_text` because it contains the extra "'s".
#   #
#   # We don't want to return `pred_text` because it's already been normalized
#   # (the SQuAD eval script also does punctuation stripping/lower casing but
#   # our tokenizer does additional normalization like stripping accent
#   # characters).
#   #
#   # What we really want to return is "Steve Smith".
#   #
#   # Therefore, we have to apply a semi-complicated alignment heruistic between
#   # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
#   # can fail in certain cases in which case we just return `orig_text`.

#   def _strip_spaces(text):
#     ns_chars = []
#     ns_to_s_map = collections.OrderedDict()
#     for (i, c) in enumerate(text):
#       if c == " ":
#         continue
#       ns_to_s_map[len(ns_chars)] = i
#       ns_chars.append(c)
#     ns_text = "".join(ns_chars)
#     return (ns_text, ns_to_s_map)

#   # We first tokenize `orig_text`, strip whitespace from the result
#   # and `pred_text`, and check if they are the same length. If they are
#   # NOT the same length, the heuristic has failed. If they are the same
#   # length, we assume the characters are one-to-one aligned.

#   tok_text = " ".join(tokenizer.tokenize(orig_text))

#   start_position = tok_text.find(pred_text)
#   if start_position == -1:
#     if  verbose_logging:
#       tf.compat.v1.logging.info(
#           "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
#     return orig_text
#   end_position = start_position + len(pred_text) - 1

#   (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
#   (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

#   if len(orig_ns_text) != len(tok_ns_text):
#     if  verbose_logging:
#       tf.compat.v1.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
#                       orig_ns_text, tok_ns_text)
#     return orig_text

#   # We then project the characters in `pred_text` back to `orig_text` using
#   # the character-to-character alignment.
#   tok_s_to_ns_map = {}
#   for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
#     tok_s_to_ns_map[tok_index] = i

#   orig_start_position = None
#   if start_position in tok_s_to_ns_map:
#     ns_start_position = tok_s_to_ns_map[start_position]
#     if ns_start_position in orig_ns_to_s_map:
#       orig_start_position = orig_ns_to_s_map[ns_start_position]

#   if orig_start_position is None:
#     if  verbose_logging:
#       tf.compat.v1.logging.info("Couldn't map start position")
#     return orig_text

#   orig_end_position = None
#   if end_position in tok_s_to_ns_map:
#     ns_end_position = tok_s_to_ns_map[end_position]
#     if ns_end_position in orig_ns_to_s_map:
#       orig_end_position = orig_ns_to_s_map[ns_end_position]

#   if orig_end_position is None:
#     if  verbose_logging:
#       tf.compat.v1.logging.info("Couldn't map end position")
#     return orig_text

#   output_text = orig_text[orig_start_position:(orig_end_position + 1)]
#   return output_text


# def _get_best_indexes(logits, n_best_size):
#   """Get the n-best logits from a list."""
#   index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

#   best_indexes = []
#   for i in range(len(index_and_score)):
#     if i >= n_best_size:
#       break
#     best_indexes.append(index_and_score[i][0])
#   return best_indexes


# def _compute_softmax(scores):
#   """Compute softmax probability over raw logits."""
#   if not scores:
#     return []

#   max_score = None
#   for score in scores:
#     if max_score is None or score > max_score:
#       max_score = score

#   exp_scores = []
#   total_sum = 0.0
#   for score in scores:
#     x = math.exp(score - max_score)
#     exp_scores.append(x)
#     total_sum += x

#   probs = []
#   for score in exp_scores:
#     probs.append(score / total_sum)
#   return probs


do_lower_case = True

null_score_diff_threshold = 0.0
version_2_with_negative = False
verbose_logging = False

max_answer_length = 30
n_best_size = 20

main()
