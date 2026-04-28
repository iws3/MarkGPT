# a file in python should end with .py
# 1. what is a variable: amything that can change, it is a memory location(container) reserve to store values for future use
# why do we need a variable: to store data, for manupulation, process data,to organise data, for code reusabilty 
# syntax of variables: name_of_variable= value_of_variable 
age=24
name= "gody";
message="Hello AI engineers";
num1=23;
num2=24;
#variable manumpulation
sum=num1 +num2;

multiplication= num1 * num2;

div= num1/num2

#how to run your code and see output
# print output
print(name)
print(age)
print(message)
# print all on thesame line
print("my name is", name, "i'm",age, "years old",message)
print("the sum is", sum )
print(multiplication)
print(div)
# variable naming conversions
# variable to string  concatination
# what is  concatination: jioning two or more strings or values togather

# program to take the users name and greet  with hello
user_name= "Gody";
greetings="Hello";
print(user_name, greetings);

# correction
userName=input("please enter your name:");
print("Hello "  + userName);