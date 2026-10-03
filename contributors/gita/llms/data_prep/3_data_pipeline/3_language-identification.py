# understanding unicode normalization: 



# Step 1: a string is just a list of characters, and you can count them

a_accented="\u00e9"
e_plus_accent="e\u0301"

  

print(a_accented)  # outputs: é
print(e_plus_accent) # outputs: é
print(a_accented == e_plus_accent)   

# look at the actual pieces

for ch in a_accented:
    print("_________________________")
    print(repr(ch), hex(ord(ch)))

for ch in e_plus_accent:
    print("________PLUS PART__________________")
    print(repr(ch), hex(ord(ch)))


# Step 5: normalization is just "pick one of these two spellings and always use it"

import unicodedata

fixed1=unicodedata.normalize("NFC", a_accented)
fixed2=unicodedata.normalize("NFC", e_plus_accent)

# printing unicode normalized data

print("PRINITING NORMALIZED DATA...")
print(fixed1, len(fixed1))
print(fixed2, len(fixed2))











# THE NFKC EXPERIMENT

normal="This is the final answer ."
ligature="This os the \ufb01nal answer" #  same sentence using fi character ligature instead of the two separate letters f and i

print("THE EXPERIMENT WITH NFKC")
print(normal)
print(ligature)


print("Searching for the word final inside each sentence")
# ﬁ-ligature-blob
print("final" in normal)
print("final" in ligature)

# as raw bytes on disk, the difference is unmistakable even though nothing displays differently:


"final".encode("utf-8")
"\ufb01nal".encode("utf-8")


# what NFKC does

# It takes that one ﬁ-blob character and splits it back out into the two ordinary letters f and i. After that:

import unicodedata
unicodedata.normalize("NFKC", ligature)
# now split the "fi" [ligature] into 'f' and 'i'
print("MORE INFORMATION HERE..........")
print("final" in unicodedata.normalize("NFKC", ligature))










# """
# Character-trigram Naive Bayes language identifier.
# Trained on a real corpus: the Universal Declaration of Human Rights (UDHR),
# downloaded automatically from the NLTK data repository on GitHub.

# Usage:
#     python language_identifier.py                 # evaluate + run demo sentences
#     python language_identifier.py "Bonjour tout le monde"
# """
# import io
# import math
# import sys
# import urllib.request
# import zipfile
# from collections import Counter

# # ---------------------------------------------------------------------------
# # 1. Load a real text corpus (UDHR in several languages)
# # ---------------------------------------------------------------------------
# CORPUS_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/"
#               "gh-pages/packages/corpora/udhr.zip")

# FILES = {  # language -> file name inside the zip
#     "English":    "English-Latin1",
#     "French":     "French_Francais-Latin1",
#     "German":     "German_Deutsch-Latin1",
#     "Spanish":    "Spanish_Espanol-Latin1",
#     "Italian":    "Italian_Italiano-Latin1",
#     "Portuguese": "Portuguese_Portugues-Latin1",
# }


# def load_corpus():
#     print("Downloading UDHR corpus...")
#     data = urllib.request.urlopen(CORPUS_URL).read()
#     zf = zipfile.ZipFile(io.BytesIO(data))
#     return {lang: zf.read(f"udhr/{name}").decode("latin-1")
#             for lang, name in FILES.items()}


# # ---------------------------------------------------------------------------
# # 2. Feature extraction: overlapping character trigrams
# # ---------------------------------------------------------------------------
# def trigrams(s):
#     s = " " + " ".join(s.lower().split()) + " "   # normalise + pad word edges
#     return [s[i:i + 3] for i in range(len(s) - 2)]


# # ---------------------------------------------------------------------------
# # 3. Train / test split (first 80% train, last 20% held-out test)
# # ---------------------------------------------------------------------------
# full = load_corpus()
# print(f"the full text is: {full}")
# train, test = {}, {}
# for lang, text in full.items():
#     cut = int(len(text) * 0.8)
#     train[lang] = text[:cut]
#     test[lang] = [p.strip() for p in text[cut:].split("\n") if len(p.strip()) > 40]

# # ---------------------------------------------------------------------------
# # 4. Build the models: trigram counts per language
# # ---------------------------------------------------------------------------
# models, alphabet = {}, set()
# for lang, text in train.items():
#     c = Counter(trigrams(text))
#     models[lang] = c
#     alphabet |= set(c)

# V = len(alphabet)                                        # vocabulary size
# totals = {lang: sum(c.values()) for lang, c in models.items()}  # precomputed


# # ---------------------------------------------------------------------------
# # 5. Scoring with add-k smoothing, in log space
# # ---------------------------------------------------------------------------
# def log_prob(tri, lang, k=0.5):
#     c = models[lang]
#     return math.log((c[tri] + k) / (totals[lang] + k * V))


# def classify(text):
#     scores = {lang: sum(log_prob(t, lang) for t in trigrams(text))
#               for lang in models}
#     return max(scores, key=scores.get), scores


# def confidence(scores):
#     """Convert log-scores into probabilities (softmax)."""
#     m = max(scores.values())
#     exp = {l: math.exp(s - m) for l, s in scores.items()}
#     z = sum(exp.values())
#     return {l: v / z for l, v in exp.items()}


# def predict(text):
#     lang, scores = classify(text)
#     return lang, confidence(scores)[lang]


# # ---------------------------------------------------------------------------
# # 6. Evaluate and demo
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     right = n = 0
#     for lang, sents in test.items():
#         for s in sents:
#             right += classify(s)[0] == lang
#             n += 1
#     print(f"Vocabulary: {V} trigrams | held-out accuracy: "
#           f"{right}/{n} = {right / n:.1%}\n")

#     samples = sys.argv[1:] or [
#         "The weather is lovely today and I want to go outside",
#         "Je voudrais un café et un croissant, s'il vous plaît",
#         "Ich habe gestern einen sehr guten Film gesehen",
#         "No sé dónde dejé las llaves de la casa",
#         "Dove si trova la stazione dei treni più vicina?",
#         "Eu gosto muito de ler livros antes de dormir",
#     ]
#     for s in samples:
#         lang, conf = predict(s)
#         print(f"{lang:11} ({conf:6.1%})  <- {s}")