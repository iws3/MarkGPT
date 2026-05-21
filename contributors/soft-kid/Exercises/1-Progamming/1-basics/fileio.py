# # # with open("data.txt", "r") as f:
# # #     content=f.read()
# # #     print(content)
# # #using the for loop with the "with" syntax
# # # with open("data.txt", "r") as f:
# # #     for line in f:
# # #         print(line)
# # #         print(line[0])

# # # how to write something into a file
# # #if you use 'w' it would over write what was already there
# # #and if you use 'a' it add to or append what was already there

# # with open("data.txt", "a") as f:
# #     content=f.write("hello world")
# #     print(content)


# # writing a csv file
# import csv  #in_built csv module
# log_data = [
#     ["timestamp", "temperature", "voltage", "current"],  # headers
#     ["09:00:00",   34.2,           220.1,     1.20],
#     ["09:05:00",   35.8,           219.5,     1.35],
#     ["09:10:00",   38.1,           221.0,     1.50],
#     ["09:15:00",   40.7,           218.8,     1.68],
# ]
# # writing the log data into a csv file using the 'w'
# # with open("sensor log.csv", "w", newline="") as file:
# #     writer = csv.writer(file)
# #     writer.writerows(log_data)
# # print("sensor log written to sensor log.csv")


# #reading a csv file
# with open("sensor log.csv", "r") as file:
#     reader = csv.DictReader(file)  # reads rows as dictionaries
#     for row in reader:
#         temp = float(row["temperature"])
#         flag = "   ← ALERT" if temp > 39 else ""
#         print(f"  {row['timestamp']} T={temp:.1f}°C{flag}")



# writing plain text (logs)
import datetime
def log_event(message, filepath ="system.log"): #filepath automatically create a file called system.log for you
    """Appenda times stamped message to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"
    with open(filepath, "a") as f:
        f.write(entry)
    print(entry, end="")
log_event("system started")
log_event("Temperature sensor online")
log_event("ALERT: Temperature threshold exceeded -41.0°C")
log_event("Emergency shutdown triggered")


