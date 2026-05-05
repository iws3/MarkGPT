# #Project 2: Electrical Calculations Toolkit 
# A command-line tool (input() for user 
# interaction) that computes Ohm's Law, Power, 
# RC time constants, and series/parallel 
# resistance. Each formula is its own function. 
# User selects which calculation to perform from a 
# menu. 
# Skills: Functions, conditionals, f-strings, while loops, user input 

#function to Calculate  ohms law when user inputs parameters known
def ohms_law():
    V = (input("Voltage: "))
    I = (input("Current: "))
    R = (input("Resistance: "))
    if V== "" and I!="" and R!="": print(f"Voltage = {float(I )* float(R):.2f} V")
    elif V!="" and I!="" and R=="": print(f"Resistance = {float(V)/float(I):.2f} \u03A9")
    elif V!="" and R!="" and I=="": print(f"Current = {float(V) / float(R):.2f} A")
    #function to Calculate  power when user inputs parameters known 
def power():
    V = (input("Voltage: "))
    I = (input("Current: "))
    R = (input("Resistance: "))
    if V!="" and I!="":    print(f"Power = {float(V )* float(I):.2f} W")
    elif I!="" and R!="":  print(f"Power = {float(I)**2 * float(R):.2f} W")
    elif V!=""and R!="":   print(f"Power = {float(V)**2 / float(R):.2f} W")

    #function to Calculate  time constant when user inputs parameters known
def RC():
     t = (input("Time: "));  R = (input("Resistance: ")); C = (input("Capacitance: "))
     if R!="" and C!="":     print(f"t = {float(R)* float(C):.2f} s")
     elif t!="" and C!="":   print(f"R = {float(t)/ float(C):.2f} u03A9") 
     elif t!="" and R!="":   print(f"C = {float(R)/ float(C):.2f} uf")

    #function to Calculate  Total resistance  when user inputs parameters known
def R_total():
     def Series():
        n =int(input("Enter resistances: "))
        Total_s= sum(float(input(f"R{i +1}: ")) for i in range(n))
        print(f"Total resistance = {Total_s:.2f} u03A9")
     def Parallel():
        m =int(input("Enter resistances: "))
        Total_s= 1/sum(1/float(input(f"R{i +1}: ")) for i in range(m))
        rt= 1/Total_s
        print(f"Total resistance = {rt:.2f} ")
     while True:
          print("\n1_series 2_parallel 3_Exit")
          choice =input("Pick: ")
          if choice == "1": Series()
          elif choice == "2": Parallel()
          else:
              break
while True:
    print("\n1_Ohms_law 2_Power 3_time_constant(RC) 4_Total_Resistance")
    choice =input("Pick: ")
    if choice == "1": ohms_law()
    elif choice == "2": power()
    elif choice == "3": RC()
    elif choice == "4": R_total()
    else:
        break
          
   
