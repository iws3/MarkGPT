import io
import math
import sys
import urllib.request
import zipfile
from collections import Counter

from requests import models

# tep A: trigrams(), the feature extractor

def trigram(s):
    s=" " + " ".join(s.lower().split()) + " "
    return [s[i:i+3] for i in range(len(s)-2)]

# ['hi', 'there'] ---> 'hi there' --> ' hi there ' --> [' hi', 'hi ', 'i t', ' th', 'the', 'her', 'ere', 're ']

# Padding with spaces is deliberate: it lets a trigram capture "this is the start of a word" (' hi') or "this is the end of a word" ('re '), not just letters floating in the middle. Then the list comprehension slides a 3-character window across the padded string:

# 8 trigrams from a 10-character padded string, matching len(s) - 2 = 8 exactly (a string of length N has N-2 windows of width 3).


# test the function:
result_test=trigram("hi there")

CORPUS_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/"
              "gh-pages/packages/corpora/udhr.zip")

FILES = {  # language -> file name inside the zip
    "English":    "English-Latin1",
    "French":     "French_Francais-Latin1",
    "German":     "German_Deutsch-Latin1",
    "Spanish":    "Spanish_Espanol-Latin1",
    "Italian":    "Italian_Italiano-Latin1",
    "Portuguese": "Portuguese_Portugues-Latin1",
}


print(result_test)

def load_corpus():
    data = urllib.request.urlopen(CORPUS_URL).read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    return {lang: zf.read(f"udhr/{name}").decode("latin-1")
            for lang, name in FILES.items()}


print("Downloading and fetching data:")

# print(load_corpus()["English"])
# print("Downloading UDHR corpus...")

train = load_corpus()

models, alphabet={}, set()

for lang, text in train.items():
    c=Counter(trigram(text))
    models[lang]=c
    alphabet |= set(c)
V=len(alphabet)
# print(f"THe alphabet is : {alphabet}")
# print(f"THE MODEL DICTIONANRY IS: {models}")
total={lang: sum(c.values()) for lang, c in models.items()}
# print(f"The totak is :{total}")
# print(f"the counter is: {c}")


# work and calculate the probability of each bigran

def log_prob(tri, lang, k=0.5):
    c=models[lang]
    return math.log((c[tri] + k) /(total[lang]+k*V))


# test the funxtin with some inputs
log_prob_test=log_prob("the", "English")
log_prob_test_french=log_prob("the", "French")
log_prob_test_german=log_prob("the", "German")
log_prob_test_spanish=log_prob("the", "Spanish")
print(f"The log probability is: {log_prob_test}")
print(f"The log probability is: {log_prob_test_french}")
print(f"The log probability is: {log_prob_test_german}")
print(f"The log probability is: {log_prob_test_spanish}")
# what does this log probality: The log probability is: -6.168666813516317 mean


