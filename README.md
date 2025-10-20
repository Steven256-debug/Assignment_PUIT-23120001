# California Housing Price Prediction Project

This project implements a machine learning model to predict median house values in California using Linear Regression. The model analyzes various features such as median income, house age, average rooms, location (latitude/longitude), and other demographic factors to predict house prices.

## Project Structure

```
california_housing_regression/
├── data/                     # Dataset storage
│   └── california_housing.csv
├── notebooks/               # Jupyter notebooks for exploration and modeling
│   ├── 01_data_exploration.ipynb
│   └── 02_model_building.ipynb
├── scripts/                # Python implementation scripts
│   ├── load_and_save_data.py
│   ├── train_linear_regression.py
│   ├── interpret_coefficients.py
│   └── visualize_results.py
├── results/                # Model outputs and visualizations
│   ├── evaluation_metrics.txt
│   ├── model_coefficients.csv
│   └── plots/
└── requirements.txt        # Python dependencies
```

## Quick Start

### Option 1: Run Everything (Recommended)

```powershell
# Run from repository root
.\run_all.ps1
```

This will:
1. Create a virtual environment
2. Install required packages
3. Download and prepare the dataset
4. Train the model
5. Generate evaluation metrics and plots

### Option 2: Manual Setup

1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r california_housing_regression/requirements.txt
```

3. Run the pipeline steps:
```powershell
python california_housing_regression/scripts/load_and_save_data.py
python california_housing_regression/scripts/train_linear_regression.py
```

## Model Performance

The current model achieves:
- R² Score: 0.6682 (explains ~66.8% of variance)
- RMSE: $66,240
- MAE: $48,230

Key feature effects (per standard deviation):
- Median Income: +$86,970
- Location (Latitude/Longitude): -$79k to -$86k
- Average Rooms: -$18,908
- House Age: +$15,058

See detailed metrics and interpretations in `california_housing_regression/results/`.

## Development

To contribute or modify:
1. Clone the repository
2. Create a virtual environment and install requirements
3. Run notebooks in `notebooks/` for exploration
4. Modify scripts in `scripts/` for implementation
5. Use `run_all.ps1` to test changes end-to-end

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn