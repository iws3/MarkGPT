# finding all prime numbers from 1 to 50
# A prime number is only divisible by 1 and itself
for j in range(2,51):
    is_prime =True
    for i in range(2, j):
       
       if j % i ==0:
          is_prime=False
          break
    if is_prime:
        print(j,  "is a prime numbers");
          

    