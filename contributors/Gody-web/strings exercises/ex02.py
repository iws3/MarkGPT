#counting of words
sentence="python is an amazing programming langauge"
words= sentence.split(" ")
number_of_words=len(words)
print(f" The number of words is {number_of_words}")

# number of characters
number_of_char=len(sentence)
print(f"the number of characters is {number_of_char}")

# number of times A apearas
A_letters=len(sentence.split("a"))
print(f"the number of As is:{A_letters}")