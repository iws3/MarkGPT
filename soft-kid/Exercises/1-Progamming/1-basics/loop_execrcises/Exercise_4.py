#solution to exercise four of loops
text = "python is great and python is easy to learn python"
words = text.split()
print(words)
counter={}
for word in words:
    if word in counter:
        counter[word]+=1
    else:
        counter[word]=1
print(counter)

