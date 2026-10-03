# tep A: trigrams(), the feature extractor

def trigram(s):
    s=" " + " ".join(s.lower().split()) + " "
    return [s[i:i+3] for i in range(len(s)-2)]

# ['hi', 'there'] ---> 'hi there' --> ' hi there ' --> [' hi', 'hi ', 'i t', ' th', 'the', 'her', 'ere', 're ']

# Padding with spaces is deliberate: it lets a trigram capture "this is the start of a word" (' hi') or "this is the end of a word" ('re '), not just letters floating in the middle. Then the list comprehension slides a 3-character window across the padded string:

# 8 trigrams from a 10-character padded string, matching len(s) - 2 = 8 exactly (a string of length N has N-2 windows of width 3).

import io
import math
import sys
import urllib.request
import zipfile
from collections import Counter
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

print(load_corpus())