#count how many times each word appears in a in text

text="python is great and python is easy to learn python"
words = text.upper().split()
startWord =[]
i =0
while i < len(words):
    word = words[i]
    if word not in startWord:
      count =0
      j =0
      while j < len(words):
        if words[j] == word:
                 count += 1
        j += 1
      print(word,":",count)
      startWord.append(word)
    i += 1
