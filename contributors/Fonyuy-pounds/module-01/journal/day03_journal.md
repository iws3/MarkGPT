# Day 03 Reflection Journal

## Exercise E03.1 — Matrix Multiplication
**Manual Calculation:**
Matrices:
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 10]] (Slightly modified to check precision)

*Calculation step for A * B:*
- Top-left: (1*5) + (2*7) = 5 + 14 = 19
- Top-right: (1*6) + (2*10) = 6 + 20 = 26
- Bottom-left: (3*5) + (4*7) = 15 + 28 = 43
- Bottom-right: (3*6) + (4*10) = 18 + 40 = 58

*Result:* [[19, 26], [43, 58]]

**NumPy Verification:**
Using `np.dot(A, B)` in [day03_solutions.py](file:///c:/Users/the%20eye%20informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-01/exercises/day03_solutions.py) confirms the manual math is correct.
Result:
```
[[19 26]
 [43 58]]
```

## Exercise E03.2 — Function Plotting
I have generated plots for the following key ML functions using [day03_solutions.py](file:///c:/Users/the%20eye%20informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-01/exercises/day03_solutions.py):

1. **Squared (f(x) = x²):** Shows the parabolic growth which is essential for understanding Mean Squared Error loss.
![Squared Function](file:///c:/Users/the%20eye%20informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-01/exercises/plots/squared_function.png)

2. **Sine (f(x) = sin(x)):** Demonstates periodicity, helpful for understanding positional encodings later in the Transformer module.
![Sine Function](file:///c:/Users/the%20eye%20informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-01/exercises/plots/sine_function.png)

3. **Sigmoid (f(x) = 1/(1+e⁻ˣ)):** Maps any real-valued number into a range between 0 and 1, used as a classic activation function and for probability scaling.
![Sigmoid Function](file:///c:/Users/the%20eye%20informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-01/exercises/plots/sigmoid_function.png)

## Exercise E03.3 — Probability Puzzle
**Calculation:**
Under a naive independence assumption: 
P("the dog barked") = P("the") * P("dog") * P("barked")
P = 0.3 * 0.1 * 0.05 = 0.0015 (or 0.15%)

**What is wrong with this assumption?**
The assumption of independence is incorrect because language is highly **dependent**. The probability of the word "dog" occurring is much higher if the previous word was "the", and the probability of "barked" is drastically higher if the subject is "dog". In real language models, we use **conditional probability** (P(w_n | w_n-1...)) rather than independent probability, because the order and relationship between words carry all the meaning.
