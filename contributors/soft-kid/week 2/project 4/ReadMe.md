## Automated CSV Explorer

# Project Overview
The Automated CSV Explorer is a command-line tool designed to provide instant insights into any dataset. By simply pointing the script at a CSV file, it automatically identifies numerical data, generates statistical summaries, and visualizes distributions—all packaged into a single, portable HTML report.

# Key Features
Dynamic Column Detection: Automatically filters and analyzes numeric columns while ignoring non-numeric data.

Statistical Profiling: Generates key metrics including mean, standard deviation, and quartiles using Pandas.

Self-Contained Reporting: Uses Base64 encoding to embed plots directly into the HTML. This means the report is a single file that can be shared via email without needing separate image attachments.

CLI Integration: Operates directly from the terminal with support for dynamic file paths.