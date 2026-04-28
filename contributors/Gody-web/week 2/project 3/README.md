# Project 3: Multi-Sensor Visualisation Report

## Overview
This script simulates 48 hours of industrial sensor data and generates a professional visual audit. It focuses on identifying correlations between environmental factors (Temperature/Humidity) and machine performance (Current/Vibration).

## Key Features
- **Realistic Data Simulation**: Variables are mathematically linked to mimic real-world physics.
- **Time-Series Dashboard**: A 4-panel grid showing every sensor's trend over 2 days.
- **Correlation Mapping**: A heatmap to identify which sensors are driving changes in others.
- **Automated HTML Reporting**: Combines all exported PNGs into a single, shareable web report.

## Technical Skills
- **Matplotlib**: Advanced subplotting and figure management.
- **Seaborn**: Heatmaps and Kernel Density Estimate (KDE) plots.
- **NumPy**: Vectorised math for simulating sensor relationships.
- **Data Export**: `savefig` for images and custom HTML generation.