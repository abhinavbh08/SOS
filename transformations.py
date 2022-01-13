import nltk
from nltk.corpus import wordnet 
import numpy as np
import math
import string
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import random

stop_words = set(stopwords.words('english'))

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)


def replace_synonym_with_wordnet(sentence, percentage = 0.3):

    tokens = nltk.word_tokenize(sentence)
    changed_tokens = tokens.copy()
    cnt = 0
    lens = len(changed_tokens)
    n = math.ceil(percentage * len([w for w in changed_tokens if (w.lower() not in stop_words and w.lower() not in string.punctuation)]))
    tries = 0
    while (cnt<n and tries<lens):
        tries += 1
        word_index = np.random.choice(lens)
        if changed_tokens[word_index].lower() in stop_words or changed_tokens[word_index].lower() in string.punctuation:
            continue
        synonyms = get_synonyms(changed_tokens[word_index])
        if len(synonyms) >= 1:
            changed_tokens[word_index] = np.random.choice(synonyms)
            cnt += 1

    changed_sentence = " ".join(changed_tokens)
    return changed_sentence

# sentences = "Actually, I preserved it bc the wording was rather nice an concise. I hope I never have to use it, but it looked different from other warnings I've seen, soI assumed it was a crafted one. Still trying to learn all the html stuffs. I have this pretty nifty idear for the April Fools front page look, but haven't the foggiest how to put together the look inside my head."
# print(replace_synonym_with_wordnet(sentences))
# from transformers import MarianMTModel, MarianTokenizer
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

def load_models():
    src_model = 'Helsinki-NLP/opus-mt-de-en'
    tgt_model = 'Helsinki-NLP/opus-mt-en-de'

    src_tokenizer = MarianTokenizer.from_pretrained(src_model)
    src_model = MarianMTModel.from_pretrained(src_model).to(device)  
    tgt_tokenizer = MarianTokenizer.from_pretrained(tgt_model)
    tgt_model = MarianMTModel.from_pretrained(tgt_model).to(device)  

    return src_tokenizer, src_model, tgt_tokenizer, tgt_model


def translate(texts, src_tokenizer, src_model, tgt_tokenizer, tgt_model, language):
    
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    # import pdb; pdb.set_trace()
    src_encoded = tgt_tokenizer.prepare_seq2seq_batch(src_texts, return_tensors='pt').to(device)  
    # import pdb; pdb.set_trace()
    tgt_encoded = tgt_model.generate(**src_encoded).to(device)  
    back_translated_decoded = src_model.generate(tgt_encoded).to(device)  
    back_translated = src_tokenizer.batch_decode(back_translated_decoded, skip_special_tokens=True)
    
    return back_translated


def back_translate(texts, src_lang="en", tgt_lang="de"):
    src_tokenizer, src_model, tgt_tokenizer, tgt_model = load_models()
    back_trans_texts = []
    batch_size = 1
    # Process in batches
    texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]    
    for text in tqdm(texts):
        back_translated_text = translate(text, src_tokenizer, src_model, 
                                        tgt_tokenizer, tgt_model, language=tgt_lang)
        back_trans_texts.extend(back_translated_text)
    return back_trans_texts

from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict = True).to(device)

def get_filled_mask(text, index):
  input = tokenizer.encode_plus(text, add_special_tokens = True, truncation = True, return_attention_mask = True, return_tensors = "pt").to(device)
  mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
  logits = model(**input)
  logits = logits.logits
  softmax = F.softmax(logits, dim = -1)
  mask_word = softmax[0, mask_index, :]
  top_word = torch.argmax(mask_word, dim=1)
  text_splitted = text.split(" ")
  text_splitted[index] = tokenizer.decode(top_word)
  return " ".join(text_splitted)


def bert_masking(sentence, percentage=0.3):

  tokens = nltk.word_tokenize(sentence)
  changed_tokens = tokens.copy()
  cnt = 0
  lens = len(changed_tokens)
  n = math.ceil(percentage * len([w for w in changed_tokens if (w.lower() not in stop_words and w.lower() not in string.punctuation)]))
  tries = 0
  input = " ".join(changed_tokens)
  output = sentence
  while (cnt<n and tries<lens):
      tries +=1
      orig_text_list = nltk.word_tokenize(output)
      random_index = random.randint(0,len(orig_text_list)-1)
      checking = orig_text_list[random_index][:]
      if checking.lower() in stop_words or checking.lower() in string.punctuation:
          continue
      orig_text_list[random_index]="[MASK]"
      mod_input = ' '.join(orig_text_list)
      output = get_filled_mask(mod_input, random_index)
      cnt+=1
  return output

# bert_masking("With these headphones, I can't hear anything.")


# text=['Here are my thoughts', 'Here are my thoughts on this thing']
# text = ["Uncle Joe gave me a red toy truck.",
# "It was no big deal.",
# "I want to get my wisdom teeth out.",
# "With these headphones, I can't hear anything.",
# "I left some donuts on the side of the road.",
# "Nothing ever really took off.",
# "The papers were mixed together in a big box.",
# "Tom wants to move into a bigger house.",
# "I love rain.",
# "It's so crowded.",
# "I want to have fun with you.",
# "The president gets a lot of criticism.",
# "Business costs will go down.",
# "You have to look closely to know.",
# "I do not like aquariums.",
# "I am a vegan.",
# "What's your favorite ice cream flavor?",
# "Who's Archibald?",
# "Could you do us a really big favor?",
# "The black lipstick was very chalky."]

# en_aug_texts = back_translate(text)
# print(en_aug_texts)

# Remarks: Not much of a change can be seen in these examples


