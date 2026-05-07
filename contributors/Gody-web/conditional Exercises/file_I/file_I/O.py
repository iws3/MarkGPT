# # #opening a file and reading or writing
# f= open("data.txt", "r")
# contain= f.read()  #doing something with file
# print(contain)
# # #closing a file
# f.close()
# with open("data.txt", "r") as f:
#     contain =f.read()
#     print(contain)
#Reading line by line with for loop
# with open("data.txt", "r") as f:
#     for line in f:
#         # print(line)
#         # print(line[1])
# # how to write file i/o
# with open("data.txt", "a") as f:
#     contain = f.write("hello,world")
    
#     print(contain)
# import csv   # Python's built-in CSV module 
# log_data = [ 
#     ["timestamp", "temperature", "voltage", "current"],
#     ["09:00:00",   34.2,             220.1,        1.20],
#     ["09:05:00",   35.8,             219.5,        1.35],
#     ["09:10:00",   38.1,             221.0,        1.50],
#     ["09:15:00",   40.7,             218.8,        1.68],
# ]

 
# # 'with' automatically closes the file — always use this pattern 
# with open("sensor_log.csv", "w",newline="") as file:
#     writer= csv.writer(file)
#     writer.writerows(log_data)

# print("sensor log written to sensor_log.csv")

 
# # ─── READING A CSV FILE
# with open("sensor_log.csv", "r") as file:
#     reader= csv.DictReader(file)
#     for row in reader:
#         current = float(row["current"])
#         temp = float(row["temperature"])
#         flag = "    <--SYSTEM NOT GOOD" if current >1.5  and temp>38.1 else "system is good"
#         print(f"{row["timestamp"]} I={current:.1f} A T={temp:.1f} °C {flag}")


        # WRITING PLAIN TEXT (logs)
import datetime
def log_event(message, filepath="system.log"):
    """Append a timestamped message to a log file."""
    timestamp= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"
    with open("filepath","a") as f:  # 'a' = append mode 
        f.write(entry)
    print(entry,end="") # also print to terminal   
log_event("sytem started")
log_event("Temperature sensor Online")
log_event("ALERT: Temperature treshold exceeds -40.1°C")
log_event("Emergency shutdown triggered")
