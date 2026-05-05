# # Smart sensor monitor
# # project to simulate 100 temperature and volage values,check for anomalies and normal temperature and voltage
# creat_sensor_csv()i
import csv
import random
# random=[]
def sensor_monitor():
    fileName ="Sensor_data.csv"
    num_row= 8
    temp_min, temp_max = 20.0, 80.0
    volt_min , volt_max= 3.0, 5.0
    rows =[["TEMPERATURE",  "VOLTAGE",   "STATUS"]]
    # print("Temp | Volt | Status")
    for i in range(1, 100):
        temp = random.uniform(20.0, 100.0)
        volt = random.uniform(3.10, 6.0)
        if temp<temp_min or temp>temp_max:
            status ="Anomalies"
            print(f"Temperature : {temp:.2f}°C  voltage: {volt:.2f} V [\033[91m{status}\033[0m]  " )
        elif volt<volt_min or volt>volt_max:
             status= "Anomalies"
             print(f"Temperature : {temp:.2f}°C  voltage: {volt:.2f} V [\033[91m{status}\033[0m]" )
        else:
           status ="Normal"
           print(f"Temperature : {temp:.2f}°C  voltage: {volt:.2f} V [\033[92m{status}\033[0m]" )
        rows.append([f"{temp:.2f} °C  " , f"  {volt:.2f} V " , status])
    with open(fileName,"w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\n{fileName} created with {num_row} rows" )
sensor_monitor()

