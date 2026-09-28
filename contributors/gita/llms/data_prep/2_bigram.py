# # # igram.py
# # import math, random
# # from collections import defaultdict, Counter
# # # wGyueHek2WxZ


# # def train_bigram(text, k=0.1):
# #     vocab=sorted(set(text))
# #     V=len(vocab)
# #     counts=defaultdict(Counter)



# # understanding counter and defaultdict

# import math, random
# from collection import defaultdict, Counter

# # initialize the nested counter
# user_actions=-defaultdict(Counter)

# logs=[
#  ('Alice', 'Login'),
#     ('Bob', 'View_Page'),
#     ('Alice', 'Click_Ad'),
#     ('Alice', 'Login')   
# ]

# # First iteration ('Alice', 'Login'): Python looks for 'Alice' in user_actions. It's not there. defaultdict instantly creates 'Alice' and assigns an empty Counter() to it. Then, that Counter registers ['Login'] += 1

# for user, action in logs:
#     user_actions[user][action]+=1

# print(user_actions)

# # output:

# # defaultdict(<class 'collections.Counter'>, {
# #     'Alice': Counter({'Login': 2, 'Click_Ad': 1}),
# #     'Bob': Counter({'View_Page': 1})
# # })

import math, random
from collections import defaultdict, Counter
def train_bigram(text, k=0.1):
    vocab=sorted(set(text))
    V=len(vocab)
    counts=defaultdict(Counter)
    for a, b in zip(text, text[1:]):
        # how many times a is followed by b
        counts[a][b]+=1;
    total={a: sum(c.values()) for a, c in counts.items()}

    # create the problaity 
    def prob(a, b):
        c=counts.get(a)
        if c is None:
            return 1.0/V
        return (c[b]+k)/(total[a]+k*V)
    # create sample
    def sample(start, n, seed=0):
        rng=random.Random(seed)
        out=[start]
        for _ in range(n):
            c=counts.get(out[-1])
            if not c:
                break
            chars, weights=zip(*c.items())
            out.append(rng.choices(chars, weights=weights)[0])
        return ''.join(out)
def avg_nll(text, prob):
    nll=sum(-math.log(prob(a,b)) for a, b in zip(text, text[1:]))
    return nll/len(text)
    # average  nll
    print(total)
# train_bigram("hello world hello everyone hello world")

text = open("input.txt", encoding="utf-8").read()
split=int(0.9 * len(text))
train, held=text[:split], text[split:]

print("Training bigram model...")
print(f"TRAIN DATA: {len(train)} characters, HELD OUT DATA: {len(held)} characters")

# Second training set: after 30% of the lines, insert a spam line.
junk = "CLICK HERE!!! BUY NOW $$$ 100% FREE www www www"

rng = random.Random(1)
noisy_lines = []
for line in train.split("\n"):
    noisy_lines.append(line)
    if rng.random() < 0.3:
        noisy_lines.append(junk)
noisy = "\n".join(noisy_lines)


print("Training bigram model on noisy data...")
print(f"SAMPLE OF NOISY DATA: {noisy[:100]}... ")


import math, random
from collections import Counter, defaultdict

def train_bigram(text, k=0.1):
    vocab = sorted(set(text))
    V = len(vocab)
    counts = defaultdict(Counter)
    for a, b in zip(text, text[1:]):
        counts[a][b] += 1
    totals = {a: sum(c.values()) for a, c in counts.items()}

    def prob(a, b):                       # add-k smoothing so nothing has probability 0
        c = counts.get(a)
        if c is None:
            return 1.0 / V
        return (c[b] + k) / (totals[a] + k * V)

    def sample(start, n, seed=0):
        rng = random.Random(seed)
        out = [start]
        for _ in range(n):
            c = counts.get(out[-1])
            if not c:
                break
            chars, weights = zip(*c.items())
            out.append(rng.choices(chars, weights)[0])
        return "".join(out)

    return prob, sample

def avg_nll(text, prob):                  # average surprise per character, in nats
    nll = sum(-math.log(prob(a, b)) for a, b in zip(text, text[1:]))
    return nll / (len(text) - 1)

text = open("input.txt", encoding="utf-8").read()
split = int(0.9 * len(text))
train, held = text[:split], text[split:]

# Second training set: after 30% of the lines, insert a spam line.
junk = "CLICK HERE!!! BUY NOW $$$ 100% FREE www www www"
rng = random.Random(1)
noisy_lines = []
for line in train.split("\n"):
    noisy_lines.append(line)
    if rng.random() < 0.3:
        noisy_lines.append(junk)
noisy = "\n".join(noisy_lines)

p_clean, s_clean = train_bigram(train)
p_noisy, s_noisy = train_bigram(noisy)

print("ln(V) =", round(math.log(len(set(text))), 3))
print("held-out loss, trained on clean:", round(avg_nll(held, p_clean), 4))
print("held-out loss, trained on noisy:", round(avg_nll(held, p_noisy), 4))
print(s_noisy("\n", 300, seed=3))
# understanding closures in python: A closure is a function object that has access to variables in its lexical scope, even when the function is called outside that scope. In Python, closures are created when a nested function references variables from its enclosing function.


# def make_adder(x):
#     def add(n):
#         return x+n
#     return add

# add4=make_adder(4)
# print(add4(5))  # Output: 9