#solution to exercise 3 of loops
# the code would find prime numbers between 1 and 50
# since one(1) is not a prime number we would start checking from 2
for num in range (2, 51):
    is_prime = True   #assume it's a prime number
    # check divisibility
    for i in range(2, num): 
        if num % i ==0:
            is_prime=False
            break
    if is_prime:
        print(num, end=" ")