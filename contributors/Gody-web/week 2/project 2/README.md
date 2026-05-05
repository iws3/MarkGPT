## 📊 Overview
The **Energy Audit Dashboard** is a Python-based data analysis tool designed to process high-frequency power meter data. This project transforms raw 5-minute interval readings into a structured daily executive summary, identifying consumption trends and peak demand periods.

## ⚙️ How It Works
1.  **Data Simulation**: The script generates 7 days of synthetic energy data, creating 288 readings per day (2,016 total data points) at 5-minute intervals.
2.  **Statistical Profiling**: Using `.describe()`, the system provides a health check of the raw sensor data.
3.  **Consumption Analysis**: 
    * Uses `groupby` to sum all intervals into a total daily kWh value.
    * Uses `resample` to isolate the exact moment (HH:MM) of maximum power draw.
4.  **Reporting**: A final table is generated, combining totals and peak times, which is then exported to a CSV file.

## 📂 File Outputs
* **`Energy_Audit_Dashboard.csv`**: A clean, formatted table containing the Date, Total kWh, and Peak Hour for the audited week.

