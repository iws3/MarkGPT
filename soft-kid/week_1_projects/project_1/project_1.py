# Project_1---------Smart Sensor monitor-------

import random
import csv
# ANSI color codes for color-coded terminal
green= '\033[92m'
red='\033[91m'
yellow = '\033[93m'
reset = '\033[0m'
# sensor normal ranges
temp_min = 20.0
temp_max = 35.0
volt_min = 3.0 
volt_max = 5.0

# counters for anomalies
anomaly_count = 0
temp_anomaly_count = 0;
voltage_anomaly_count = 0;

# list to store readings
all_reading = []

def color_coded_terminal_report():
    print(f"{yellow}====Smart Sensor Monitor===")
    print(f"Normal ranges: temp {temp_min} to {temp_max} in °C|| volt {volt_min} to {volt_max} in volts(V)")
    print()
    print(f"{'ID':<4} {'Temp (°C)':<12} {'voltage (V)':<12} {'status':<10}")
    print("-" * 40)

def generating_realistic_values():
    temp = random.normalvariate(25.0, 2.5)
    volt= random.normalvariate(4.0, 0.4)
    return temp, volt

def anomalies_detector(temp, volt, reading_id):
    is_temp_anomaly = temp < temp_min or temp > temp_max
    is_volt_anomaly = volt < volt_min or volt > volt_max

    if is_temp_anomaly:
        global temp_anomaly_count
        temp_anomaly_count+= 1
    if is_volt_anomaly:
        global voltage_anomaly_count
        voltage_anomaly_count+= 1
    if is_temp_anomaly or is_volt_anomaly:
        global anomaly_count
        anomaly_count += 1
        status = "ANOMALY"
        # status=f"{red}ANOMALY{reset}"
    else:
        status = "NORMAL" 
    # rounding values for clean display and logging
    temp_display = round(temp, 2)
    volt_display = round(volt, 2)

    # storing all values in a list
    current_reading = [reading_id, temp_display, volt_display, status ]
    # for reading in current_reading:
    all_reading.append(current_reading)

def simulating_100_readings():
    for reading_id in range(1, 101):
        temp, volt = generating_realistic_values()
        # introducing random anomalies at a 15% chances
        if random.random() < 0.15:
            if random.random() < 0.5:
                temp = random.uniform(10.0, 19.9) 
            else:
                temp=random.uniform(35.1, 45.0)
        if random.random() < 0.15:
            if random.random() < 0.5:
                volt = random.uniform(2.0, 2.9)
            else:
                volt = random.uniform(5.1, 6.0)
    
        anomalies_detector(temp, volt, reading_id)
        
#logging
with open("sensor_data.csv",  "w", newline="") as file:
    writer= csv.writer(file)
    writer.writerow(['Reading_id', 'Temperature_C', 'Voltage_V',  'Status'])
    color_coded_terminal_report()
    simulating_100_readings()
    writer.writerows(all_reading)
    
    #Final summary
    print("-"*40)
    print(f"{yellow}SUMMARY REPORT{reset}")
    print("Total readings simulated: 100")
    print(f"Total anomalies detected : {red if anomaly_count > 0 else green}{anomaly_count}{reset}")
    print(f" -Temperature anomalies : {temp_anomaly_count}")
    print(f" -Voltage anomalies : {voltage_anomaly_count}")
print(all_reading)
    


    