# solution to exercise three of conditionals
correct_username ="admin"
correct_password = "password123"
username =input("Enter Username: ")
password =input("Enter password: ")
if username == correct_username and password==correct_password:
    print("Access granted")
    print("Welcome home")
else:
    print("Access denied....")
    print("Enter correct password or username")
