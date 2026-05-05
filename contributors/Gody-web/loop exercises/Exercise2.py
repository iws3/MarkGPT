# solution to pattern printing

row= 1;   
while row <=5:
    for star in range(row):
        print("*", end=" ");
    row +=1;
    print();
row=4
while row >=1:
    for star in range(row):
        print("*", end=" ");
    row -=1;
    print(); 