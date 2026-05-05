# Calculation of all numbers from 1 to 100
# ADDITION TIME TABLE

# 1. using for loop to calculate
for i in range(1,100):
    for j in range(1,100):
        sum= i+j;
    #     print(i, "+" ,j, "=", sum);
    # print();

# 2. Using while loop
number=1;
while number <100:
    for number1 in range(1,100):
        for number2 in range(1,100):
             sum= number1 +number2;
             print(number1, "+" ,number2, "=", sum);
    number+=1;