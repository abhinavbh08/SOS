import nltk
from nltk.corpus import wordnet 
import numpy as np
import math
import string
from nltk.corpus import stopwords
from tqdm import tqdm

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