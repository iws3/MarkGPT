# Project 1: Sensor Data Analyser

## Overview
This project simulates an industrial multi-channel sensor system using **NumPy**. It generates 10,000 synthetic readings for **Temperature**, **Current**, and **Voltage** to model a real-world power monitoring system. The tool performs high-speed statistical analysis and automatically extracts sensor anomalies for further inspection.

## Features
* **Realistic Simulation:** Generates 10,000 data points using `numpy.random`.
* **System Modeling:** Uses a linear physics model ($V = 230 - 0.5 \times I$) with added Gaussian noise to simulate voltage drop across a load.
* **Vectorized Processing:** Leverages NumPy's vectorization to compute means and standard deviations across all three channels simultaneously.
* **Anomaly Detection:** Uses **Boolean Indexing** to identify sensor data that falls outside of the expected operating range (Statistical Outliers).
* **Data Export:** Efficiently saves flagged "Sensor Anomalies" to a CSV file using `np.savetxt`.
README.md
## Sensor ChannelsREADME.md
| Channel | Unit | Description |
| :--- | :--- | :--- |README.md
| **Temperature** | °C | Ambient operating temperature of the system. |
| **Current** | A | Amperage load including sensor measurement noise. |
| **Voltage** | V | Resulting line voltage after resistance-based drop. |