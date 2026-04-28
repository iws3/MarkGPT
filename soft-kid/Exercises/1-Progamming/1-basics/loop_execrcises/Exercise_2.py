# solution of exercise two of loops
for row in range(1, 5):
    for star in range(row):
        print("*", end="")
    print() # New line after each row
for row_2 in range(5, 0, -1):
    for star in range(row_2):
        print("*", end="")   
    print()  # New line after each row