# Project 2: Electrical calculation toolkit
# menu
def menu():
    print("\n==== CALCULATION MENU=====")
    print("1. Ohm's law calculations")
    print("2. Power calculations")
    print("3. RC time constant calculations")
    print("4. Series/Parallel resistnce calculations")
    print("5. exit")

# ohms law calcualtions
def Ohms_law():
    I, R = map(float, input("Enter the values of current and resistance respectively: ").split())
    V = I * R
    print(f"The voltage is: {V:.2f} volts(V)")

#power calculations
def power():
    V, I = map(float, input("Enter your voltage and current values respectively: ").split())
    P = V * I
    print(f"The power of {V}V and {I}I is {P:.4f} watts(W)")

# RC time constant calcuations
def RC_time_constant():
    C, R = map(float, input("Enter the values of capacitance and resistance respectively: ").split())
    tau = R * C
    print(f"Time constant τ = {tau:.1f} sec(S)")

# series and parallel resistance
def resistance():
    R_1, R_2 = map(float, input("Enter two resistance values respectively: ").split())

    #series connection
    RS = R_1 + R_2
    print(f"The series total resistance is: {RS} ohms")

    # parallel connection
    Rp = (R_1 * R_2)/(R_1 + R_2)
    print(f"The parallel total resistance is: {Rp} ohms")

def display():
    while True:
        menu()
        choice = int(input("Select an option (1-5): "))
        if choice == 1:
            Ohms_law()
        elif choice == 2:
            power()
        elif choice == 3:
            RC_time_constant()
        elif choice == 4:
            resistance()
        elif choice == 5:
            break
        else:
            print("Invalid option.......please try again")

display()   