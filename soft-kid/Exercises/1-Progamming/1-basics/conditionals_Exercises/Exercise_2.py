# solution to exercise two of conditionals
year= int(input("Enter year: "))
if year % 4 == 0 and year % 100 != 0:
    print(f"{year} is a leap year")
else:
    print(f"{year} is NOT a leap year")