# Leap year exercise solution
year = int (input("Enter your year to check if it's a Leap year: "));
if year%4==0 and year%100 !=0 or year%400==0:
    print("LEAP YEAR")
else:
    print("NOT A LEAP YEAR");
