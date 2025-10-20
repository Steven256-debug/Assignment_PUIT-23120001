# Data manipulation and numerical operations
import pandas as pd
import numpy as np
from pathlib import Path

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# System utilities
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def create_dirs():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / 'results'
    plots_dir = results_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, results_dir


def load_data():
    """Load and preprocess the dataset."""
    try:
        base_dir = Path(__file__).resolve().parent.parent
        data_path = base_dir / 'data' / 'california_housing.csv'
        df = pd.read_csv(data_path)
        print("✓ Data loaded successfully from", data_path)
        return df
    except FileNotFoundError:
        print("❌ Error: Data file not found. Please run scripts/load_and_save_data.py first from the project root or ensure data/california_housing.csv exists.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit(1)


def handle_outliers(df, columns, lower_percentile=1, upper_percentile=99):
    """Handle outliers by capping values at specified percentiles."""
    df_copy = df.copy()
    for col in columns:
        lower = np.percentile(df_copy[col], lower_percentile)
        upper = np.percentile(df_copy[col], upper_percentile)
        df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
    return df_copy


def main():
    # Create directories
    base_dir, results_dir = create_dirs()

    # Load and preprocess data
    df = load_data()
    # Handle outliers
    numeric_cols = ['Median_Income', 'House_Age', 'Average_Rooms', 'Average_Bedrooms',
                    'Population', 'Average_Occupancy', 'Median_House_Value']
    df = handle_outliers(df, numeric_cols)
    print("✓ Outliers handled")

    # Prepare features and target
    X = df.drop('Median_House_Value', axis=1)
    y = df['Median_House_Value']

    # Create train/test split with stratification
    try:
        y_quartiles = pd.qcut(y, q=4, labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_quartiles
        )
        print("✓ Data split completed")
    except Exception as e:
        print(f"❌ Error splitting data: {str(e)}")
        exit(1)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled")

    # Train model with cross-validation
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")

    # Train final model on full training data
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics
    metrics_path = results_dir / 'evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f'Mean Squared Error: {mse:.4f}\n')
        f.write(f'Root Mean Squared Error: {rmse:.4f}\n')
        f.write(f'Mean Absolute Error: {mae:.4f}\n')
        f.write(f'R-squared: {r2:.4f}\n')
        f.write(f'\nCross-validation mean score: {cv_scores.mean():.4f}\n')

    # Save coefficients with feature importance ranking
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)

    coefficients.to_csv(results_dir / 'model_coefficients.csv', index=False)
    print('\n✅ Model training complete.')
    print(f'✅ Results saved to {results_dir}')


if __name__ == '__main__':
    main()
