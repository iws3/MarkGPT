# Project 4: Automated CSV Explorer

## Description
A command-line tool that performs instant exploratory data analysis (EDA) on any CSV file. It automatically identifies numerical data, calculates key statistics, and visualizes distributions.

## Features
- **Dynamic Input**: Accepts any local CSV path via terminal arguments.
- **Auto-Filtering**: Automatically separates numeric data from strings/objects.
- **Single-File Portability**: Uses Base64 encoding to embed plots directly into the HTML, making the report easy to share via email.
- **Modern UI**: Styled with Bootstrap for a clean, professional look.

## Requirements
- Pandas
- Matplotlib

## Usage
1. Open CMD or Terminal.
2. Navigate to the script directory.
3. Run: `python explorer.py your_dataset.csv`