import sys
import csv

# 1. SETUP: Name of the file we want to read
data_file = "Sensor_data.csv" 

# 2. BUCKETS: Empty lists to store our numbers

alarm_count = 0

# 3. READ FILE: Open the folder and look inside
with open(data_file, "r") as f:
    # Read all lines but skip the top header row
    reader =csv.DictReader(f) 
    lines = f.readlines()[1:]
    all_temps = []
    all_volts = []
    
    for line in lines:
        # Remove invisible spaces and skip empty lines
        clean_line = line.strip()
        if not clean_line:
            continue
            
        # Cut the line into pieces at every comma
        parts = clean_line.split(",")
        
        # If the line has all 4 pieces of info, save them
        if len(parts) >= 4:
            # Change text into numbers
            t = float(parts[1])
            v = float(parts[2])
            status = parts[3]
            
            # Put numbers into our buckets
            all_temps.append(t)
            all_volts.append(v)
            
            # Count the alarms
            if status == "ALARM":
                alarm_count += 1

# 4. MATH: Calculate totals and averages
total = len(all_temps)

if total > 0:
    avg_t = sum(all_temps) / total
    avg_v = sum(all_volts) / total

    # 5. REPORT: Print the dashboard to the screen
    print("\n" + "="*40)
    print(f" DASHBOARD: {data_file}")
    print("="*40)
    print(f"Total Records: {total}")
    print(f"Total Alarms:  {alarm_count}")
    print("-" * 40)

    # Show Min, Max, and Average
    print(f"TEMP | Min: {min(all_temps):.1f} | Max: {max(all_temps):.1f} | Avg: {avg_t:.1f}")
    print(f"VOLT | Min: {min(all_volts):.1f} | Max: {max(all_volts):.1f} | Avg: {avg_v:.1f}")

    # 6. CHART: Visual bar for hot readings (>80)
    hot_ones = len([x for x in all_temps if x > 80])
    print(f"\nHot Readings: {'#' * hot_ones} ({hot_ones})")
    print("="*40)
else:
    print("Error: No data found inside the file!")