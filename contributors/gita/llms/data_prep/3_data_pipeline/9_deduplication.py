import re
def shingles(text, n=5):
    words=re.findall(r"\w+", text.lower())
    print(f"Words: {words}")
    if len(words)<=n:
        return {" ".join(words)}
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}

def jaccard(a, b):
    return len(a&b)/len(a|b) if (a or b) else 0.0

# Sample texts with slight differences
text1 = "The quick  brown fox jumps over the lazy dog!"
text2 = "A quick brown fox jumps over that lazy dog."

# 1. Extract shingles
shingles1 = shingles(text1, n=5)
shingles2 = shingles(text2, n=5)
print(f"Shingles for Text 1: {shingles1}")
print(f"Shingles for Text 2: {shingles2}")

# 2. Calculate similarity
similarity = jaccard(shingles1, shingles2)

print(f"Text 1 Shingles: {shingles1}\n")
print(f"Text 2 Shingles: {shingles2}\n")
print(f"Jaccard Similarity: {similarity:.2f}")
