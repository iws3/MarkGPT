import csv
import random
from datetime import datetime, timedelta

# 1. THE THERMOMETER: This function creates one temperature reading
def read_temperature(ts):
    # Calculates a base temp that peaks at 4:00 PM (16:00)
    h = ts.hour + ts.minute / 60
    base = 18 + 10 * (1 - abs(h - 16) / 16) 
    temp = round(base + random.uniform(-1.0, 1.0), 2)
    
    # Label the data: Is it safe (Normal) or too hot/cold?
    status = "NORMAL" if 20.0 <= temp <= 28.0 else "OUT_OF_RANGE"
    
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "temp_c": temp,
        "status": status
    }

def main():
    # 2. THE TIMELINE: Create 288 times (every 5 mins for 24 hours)
    start = datetime.now().replace(hour=0, minute=0, second=0)
    times = [start + timedelta(minutes=i * 5) for i in range(288)]
    
    # 3. GENERATE DATA: Create all 288 records using our function
    logs = [read_temperature(t) for t in times]
    
    # 4. SAVE TO FILE: Write the list into a CSV spreadsheet
    with open("sensor_monitor.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "temp_c", "status"])
        writer.writeheader()
        writer.writerows(logs)

    # 5. CALCULATE HOURLY AVERAGES
    print(f"\n{'HOUR':<8} | {'AVG TEMP':<10} | {'RECORDS'}")
    print("-" * 35)

    # We loop 24 times (once for every hour)
    for h in range(24):
        # We grab 12 readings at a time (12 readings * 5 mins = 60 mins)
        # This is called 'slicing' a list
        hour_slice = logs[h*12 : (h+1)*12]
        
        # Pull out just the temperature numbers from that hour
        temps_only = [row["temp_c"] for row in hour_slice]
        hourly_avg = sum(temps_only) / len(temps_only)
        
        # Print a neat row in our table
        print(f"Hour {h:02}:00 | {hourly_avg:>8.2f}°C | {len(temps_only)} logs")

    print("-" * 35)
    print("Done. Check 'sensor_monitor.csv' for the full list.")

# Run the program
main()