# KNN Project

An end-to-end Streamlit application for K-Nearest Neighbors (KNN) classification and regression. The app supports dataset download or CSV upload, EDA, data cleaning, model training, evaluation, cross-validation, and k-value analysis.

## Features
- Download sample datasets (Iris, Diabetes, Wine/Penguins) or upload your own CSV
- Exploratory Data Analysis (EDA) with summaries and visualizations
- Missing value handling and duplicate removal
- Save and load cleaned datasets
- KNN classification or regression with optional scaling and grid search
- Performance metrics, plots, and cross-validation

## Project Structure
```
app.py
data/
  raw/
  cleaned/
```

## Setup
1. Create and activate a Python environment (optional but recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run the App
```
streamlit run app.py
```

## Notes
- Cleaned datasets are saved to the data/cleaned folder with a timestamped filename.
- Raw datasets are stored in data/raw.
