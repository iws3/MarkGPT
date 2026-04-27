# Project 1: Smart Sensor Monitor

This project simulates 100 temperature and voltage values. It detects and counts anomalies (values outside normal bounds). Prints a colour-coded terminal report. Logs all readings and flagged events to `sensor_data.csv`.

## Features
- Simulates 100 readings of temperature and voltage using `random`
- Detects anomalies: values outside normal bounds
- Counts total number of anomalies
- Prints colour-coded report to terminal
- Logs all readings + flagged events to `sensor_data.csv`

## Files
| File | Description |
| --- | --- |
| `sensor_monitor.py` | Main script for simulation, anomaly detection, and logging |
| `sensor_data.csv` | CSV log of all 100 readings with anomaly flags |
