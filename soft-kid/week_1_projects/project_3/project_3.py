# =========DATA LOGGER SIMULATOR==============
# project case study is "TEMPERATURE"
import csv
import random
from datetime import datetime, timedelta

start_time = datetime(2026, 4, 20, 0, 0, 0)
sensor_data=[]


# generating realistic temperature values
def generating_realistic_values():
    for i in range(288):
        temp = round(random.uniform(22, 27), 2)
        return temp

#loop through the 288 records for the 24 hours, every 5 min
def record_loop(temp):
    for i in range (1, 289):
        # adding time every after 5min for 24 hours
        current_time= start_time + timedelta(minutes=5 * i)
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        sensor_data.append(timestamp_str)
        sensor_data.append(temp)
    
temp = generating_realistic_values()
record_loop(temp)

#writing to csv file
def csv_write_file():
    with open("sensor_log.csv", "w", newline="") as file:
        writer=csv.writer(file)
        writer.writerow(['timestamp', 'temperature_value_in_°C'])
        for i in range (0, len(sensor_data), 2):
            if i + 1 < len(sensor_data):
                writer.writerow([sensor_data[i], sensor_data[i + 1]])
csv_write_file()

#Reading back sensor data
hourly_data= {hour: [] for hour in range (24)}
def csv_read_file():
    with open("sensor_log.csv", "r") as file:
        reader = csv.DictReader(file)
        # next(reader) #skips header
        for row in reader:
            # print(row)
            time = "".join(row["timestamp"])
            dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            hour = dt.hour
            hourly_data[hour].append(float(row["temperature_value_in_°C"]))

csv_read_file()


# computing the hourly average
hourly_averages= {}
def hourly_average():
    for hour in range (24):
        values= hourly_data[hour]
        avg=sum(values)/len(values)
        hourly_averages[hour] = round(avg, 2)
hourly_average()

# printing the hourly stats

print("HOURLY AVARAGES TABLE FOR TEMPERATURE IN °C")
print("+-------+-----------------+")
print("| HOURS |   Avg Readings  |")
print("+-------+-----------------+")

for hour in range(24):
    avg=hourly_averages[hour]
    print(f"| {hour:02d}    |  {avg:12.2f}   |")
print("+-------+-----------------+") 