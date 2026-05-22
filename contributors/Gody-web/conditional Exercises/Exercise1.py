# solution to conditional Exercise 1
score=int(input("Enter your score: "));
if score >= 90 and score <= 100:
    Grade = "A";
    # invalid="ivalid score, enter the right score"
    print("your Grade is: ", Grade );
elif score >=85 and score<= 89:
    Grade="B+"
    print("your grade is: ", Grade);
elif score >=80 and score<= 84:
     Grade="B-"
     print("your grade is: ", Grade)
elif score >=75 and score<= 79:
    Grade="C+"
    print("your grade is: ", Grade);
elif score >=70 and score<= 74:
     Grade="C-"
     print("your grade is: ", Grade)
elif score >=65 and score<= 69:
    Grade="D+"
    print("your grade is: ", Grade);
elif score >=60 and score<= 64:
     Grade="D-"
     print("your grade is: ", Grade)
elif score >= 0 and score <= 60:
    Grade="F"
    print("your grade is: ", Grade);
else:
     Grade="INVALID SCORE, PLEASE ENTER THE CORRECT SCORE"
     print(Grade)