# UK Electricity Demand Forecasting
### MSc Data Science Dissertation - Manchester Metropolitan University 2026

## Overview
This project implements and compares five forecasting approaches for UK national electricity demand using 15 years of half-hourly settlement period data (2009-2024). MSc Data Science dissertation at Manchester Metropolitan University, supervised by Dr Philip Sinclair.

## Results
All models evaluated on daily-aggregated predictions for a fair comparison:

| Model | RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|---|
| XGBoost (Tuned) | 78.58 | 57.33 | 0.23 |
| XGBoost (Default) | 95.43 | 73.48 | 0.29 |
| GRU | 201.73 | 158.75 | 0.59 |
| LSTM | 366.25 | 277.08 | 1.09 |
| Hybrid ARIMA+XGBoost | 3818.22 | 3118.09 | 11.94 |
| ARIMA | 4088.88 | 3321.19 | 12.64 |

## Key Findings
- XGBoost (tuned) achieves best performance (MAPE 0.23%), driven by strong short-term autocorrelation where lag_1 accounts for ~70% of feature importance
- GRU outperforms LSTM with lower variance across runs (1.81 +/- 0.19% vs 3.90 +/- 1.10%)
- Hybrid ARIMA+XGBoost underperforms standalone models due to ARIMA's inability to capture sub-daily seasonal variation

## Dataset
UK National Electricity Consumption 2009-2024 (Vidalrod, 2024)
- 279,264 half-hourly national demand readings
- Download from: https://www.kaggle.com/datasets/albertovidalrod/electricity-consumption-uk-20092022
- Note: Raw data not included in repo due to size. Download and place historic_demand_2009_2024.csv in project root.

## Project Structure
- Project.ipynb - Main analysis notebook (preprocessing, all models, evaluation)
- dashboard.py - Interactive Streamlit dashboard
- requirements.txt - Python dependencies
- Output/ - Saved model weights, plots, and results CSVs

## Models
- ARIMA(7,1,1) - statistical baseline on daily data
- XGBoost - gradient boosted trees with 16 lag and temporal features, tuned via RandomizedSearchCV
- LSTM - 2-layer PyTorch network, look-back 48 periods, early stopping
- GRU - 2-layer PyTorch network, look-back 48 periods, early stopping
- Hybrid ARIMA+XGBoost - ARIMA trend + XGBoost residual correction

## Installation
1. Clone the repo
2. pip install -r requirements.txt
3. Download dataset from Kaggle and place in project root
4. Run: jupyter notebook Project.ipynb
5. Dashboard: streamlit run dashboard.py

## Author
Prashant Kumar Bhalla
MSc Data Science, Manchester Metropolitan University 2026
Student ID: 25948685
Supervisor: Dr Philip Sinclair
