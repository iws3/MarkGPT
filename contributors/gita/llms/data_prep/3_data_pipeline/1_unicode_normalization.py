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









