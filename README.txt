
# Project: Fractal Dimension-Based Analysis of Evolutionary Trends in Academic Research

This project explores time series, sentiment, and topical analysis of academic research using fractal dimensions. The analysis process relies on multiple Python scripts, and it is essential to follow the correct order of execution.

## Dataset

The original dataset is stored in `arxivData.json`. This dataset must first be processed and cleaned before any further analysis can be done. The cleaning process will convert the `.json` data into an Excel file, which is required by the analysis scripts.

Prerequisites

Ensure that the following are installed:

- Python 3.x
- Pandas
- NumPy
- SciPy
- Sklearn
- Matplotlib
- Any other dependencies listed in `requirements.txt`

Project Files

- `arxivData.json`: The original dataset used for this project.
- `DataCleaningCode.py`: The script that processes the `arxivData.json` file and generates the required Excel file.
- `SentimentalAnalysis.py`: The script that performs sentiment analysis on the cleaned data.
- `TimeSeriesAnalysis.py`: The script for time series analysis based on the cleaned data.
- `TopicModellingAnalysis.py`: The script that performs topic modeling and analysis using fractal dimensions.
  
Workflow

1. Step 1: Data Cleaning
   
   The first step in the process is to run the `DataCleaningCode.py` script, which takes `arxivData.json` and generates a cleaned Excel file. This Excel file will be used by the subsequent analysis scripts.


   This step is mandatory before running any analysis code.

2. Step 2: Sentiment Analysis

   Once the data is cleaned, you can proceed with sentiment analysis. Run the `SentimentalAnalysis.py` script:


3. Step 3: Time Series Analysis

   For time series analysis of the data, run the `TimeSeriesAnalysis.py` script:


4. Step 4: Topic Modelling Analysis

   To perform topic modeling and uncover hierarchical structures within the data, run the `TopicModellingAnalysis.py` script:


Notes

- Ensure that `DataCleaningCode.py` is run before any analysis scripts to generate the required Excel file.
- Each analysis script is dependent on the cleaned data from the previous step, so following the workflow order is crucial.


