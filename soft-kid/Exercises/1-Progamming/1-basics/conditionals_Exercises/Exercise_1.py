# solution to exercse one of conditionals
#print("Enter the scores: ") 
score= int(input("Enter the scores: "))
if score>=97 and score<=100:
    grade="A+"
elif score>=90 and score<=96:
    grade="A"
elif score>=87 and score<=89:
    grade="B+"
elif score>=83 and score<=86:
    grade="B-"
elif score>=80 and score<=82:
    grade="B"
elif score>=77 and score<=79:
    grade="C+"
elif score>=73 and score<=76:
    grade="C-"
elif score>=70 and score<=72:
    grade="C"
elif score>=67 and score<=69:
    grade="D+"
elif score>=63 and score<=66:
    grade="D-"
elif score>=60 and score<=62:
    grade="D"
elif score>=0 and score<60:
    grade="F"
else:
    grade="Please enter correct scores"
print(f"Your grade is: {grade}")