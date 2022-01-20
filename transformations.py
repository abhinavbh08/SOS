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


def delete_chars(sentence, percentage = 0.3):

    tokens = nltk.word_tokenize(sentence)
    changed_tokens = tokens.copy()
    cnt = 0
    lens = len(changed_tokens)
    n = math.ceil(percentage * len([w for w in changed_tokens if (w.lower() not in stop_words and w.lower() not in string.punctuation)]))
    tries = 0
    used_index = []
    while (cnt<n and tries<lens):
        tries += 1
        word_index = np.random.choice(lens)
        if word_index in used_index:
            continue
        if changed_tokens[word_index].lower() in stop_words or changed_tokens[word_index].lower() in string.punctuation:
            continue
        word = changed_tokens[word_index]
        idx = np.random.choice(len(word))
        used_index.append(word_index)
        len_wrd = len(word)
        if idx == 0:
            word = word[1:]
        elif idx == (len_wrd-1):
            word = word[:idx]
        else:
            word = word[:idx] + word[(idx+1):]

        changed_tokens[word_index] = word

    changed_sentence = " ".join(changed_tokens)
    return changed_sentence


# delete_chars("This is a nice movie and good result")

# sentences = "Actually, I preserved it bc the wording was rather nice an concise. I hope I never have to use it, but it looked different from other warnings I've seen, soI assumed it was a crafted one. Still trying to learn all the html stuffs. I have this pretty nifty idear for the April Fools front page look, but haven't the foggiest how to put together the look inside my head."
# print(replace_synonym_with_wordnet(sentences))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
  while (cnt<n and tries<min(lens, 512)):
      tries +=1
      orig_text_list = nltk.word_tokenize(output)
      random_index = random.randint(0,len(orig_text_list)-1)
      checking = orig_text_list[random_index][:]
      if random_index > 300:
          continue
      if checking.lower() in stop_words or checking.lower() in string.punctuation:
          continue
      orig_text_list[random_index]="[MASK]"
      mod_input = ' '.join(orig_text_list)
      output = get_filled_mask(mod_input, random_index)
      cnt+=1
  return output

# bert_masking("'== MyUID Cards ==    Do you know that because of uniqueness UID card can be India’s first multi utility smart card. Because of these cards your life will be easy and govt. can control & monitor on corruption. It can become your PAN Card, Voter I Card (Election Card), Driving license, Ration Card, employment card, Debit card, Credit card. it can be enroll in other center & state scheme. There is another benefits of that today financial companies are not providing loans in negative areas but after UID may possible they can provide loan in negative area because bank will easily get the details of defaulters & common man will really get the benefits of UID.    There is lots of other uses of UID card like controlling on corruption, cash subsidy, genetic profile, health card, all degree & school certificate details will be there with Grade,  . As we know that the corruption is the major road block of our country. Now the “Bhrashtachaar has become a Shishtachaar”. We need to hit on it because now corruption’s flow is equal to our GDP.          Know about UIDAI Authority & its projects.    The Unique Identification Authority of India (UIDAI) (Hindi: भारतीय विशिष्ट पहचान प्राधिकरण), is an agency of the Government of India responsible for implementing the envisioned Multipurpose National Identity Card or Unique Identification card (UID Card) project in India. It was established in February 2009, and will own and operate the Unique Identification Number database. The authority will aim at providing a unique number to all Indians, but not smart cards. The authority would provide a database of residents containing very simple data in biometrics.   The agency is headed by a chairman, who holds a cabinet rank. The UIDAI is part of the Planning Commission of India. Mr. Nandan Nilekani, a former co-chairman of Infosys Technologies, was appointed as the first Chairman of the authority in June 2009.Mr. Ram Sewak Sharma, an IAS Officer of Jharkhand Government cadre has been appointed as the Director General and Mission Director of the Authority. He is known for his best effort in e-Governance project for Jharkhand State and working as an IT secretary he received a number of awards for best Information Technology Trends State in India.    Launch    UIDAI launched AADHAAR program in the tribal village, Tembhli, in Shahada, Nandurbar, Maharashtra on 29th September, 2010. The program was inaugurated by Prime Minister, Manmohan Singh along with UPA chairperson Sonia Gandhi. The first resident to receive an AADHAAR was Rajana Sonawane of Tembhli village.    UIDAIcards.com is providing a platform to express your views, thoughts and suggestions for the success of UID project… Our best wishes and support will be with UID authority 24×7 for the success of UID Project…    What we think about the UID card projects and how it will be use.    My UID Card, My New UID Card, UID Card in Delhi, ID Card Delhi, UID card application form, UID Card Format, UID card details, UID card Registration, UID Card Cost, UID card Status, UID Card Registration Form, HOW To UID CARD, UID Card, UID Card India, US security card, America Security card, UK Security card, USA UID Card, UK UID Card, Social Security card, UIDAICARD.com, myuidcard.com, myuidcard.in, myuidcard.co.in       My UID Card, My New UID Card, HOW To UID CARD, UID Card, UID Card India, US security card, America Security card, UK Security card, USA UID Card, UK UID Card, Social Security card, UIDAICARD.com, myuidcard.com, myuidcard.in, myuidcard.co.in         Card’s configuration should be very simple to know person’s location without help of any device   The card should be available   in form of smart card.   (See the demo card)       Code       #    Use for Unique Identity cards (Indians)   *    Use for NRI.   //    Use for foreigners working in embassies or Ministers/ Officials.   =    Use for foreign visitors in India.   */    Foreigners are working in Indian.   0xxx    Country code for NRI where he is living now..   0011    Delhi (state code) NRI belongs to Delhi state   0000    NRI belongs to particular district.   **** **** **** The 12 digits will belongs to UID Number    Card’s numbers should like it    # 0091 0011 0000 **** **** **** for Indian    * 0001 0212 0000 **** **** **** for NRI (US Citizen)    // 0001 0212 0000 **** **** **** for other foreigners working in India.    = 0001 0011 0000 **** **** **** for foreign visitors.    */ 0001 0032 0000 **** **** **** for foreigners working in India.   How the card can be multipurpose for us:            * Debit Card (Before transaction the person has to enter its last four digits of virtual Debit cards)       * Credit Card ('")




