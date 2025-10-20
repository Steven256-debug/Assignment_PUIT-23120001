# Data processing and numerical operations
import pandas as pd
import numpy as np

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# System utilities
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def create_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../results/plots', exist_ok=True)


def load_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv('../data/california_housing.csv')
        print("✓ Data loaded successfully")
        return df
    except FileNotFoundError:
        print("❌ Error: Data file not found. Please run load_and_save_data.py first.")
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


def train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate multiple models, return the best one."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }

    best_model = None
    best_score = float('-inf')
    best_name = None

    for name, model in models.items():
        # Train and evaluate with cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mean_cv_score = cv_scores.mean()

        print(f"\n{name}:")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {mean_cv_score:.4f}")

        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
            best_name = name

    # Train the best model on full training data
    best_model.fit(X_train_scaled, y_train)
    return best_model, best_name, best_score


def main():
    try:
        # Create directories
        create_dirs()

        # Load and preprocess data
        df = load_data()

        # Handle outliers
        numeric_cols = ['Median_Income', 'House_Age', 'Average_Rooms', 'Average_Bedrooms',
                        'Population', 'Average_Occupancy', 'Median_House_Value']
        df = handle_outliers(df, numeric_cols)
        print("✓ Outliers handled")

        # Add engineered features
        df['Rooms_per_Person'] = df['Average_Rooms'] / df['Average_Occupancy']
        df['Bedrooms_Ratio'] = df['Average_Bedrooms'] / df['Average_Rooms']
        print("✓ Feature engineering completed")

        # Prepare features and target
        X = df.drop('Median_House_Value', axis=1)
        y = df['Median_House_Value']

        # Create train/test split with stratification
        y_quartiles = pd.qcut(y, q=4, labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_quartiles
        )
        print("✓ Data split completed")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✓ Features scaled")

        # Train and evaluate models
        best_model, best_name, best_score = train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Get predictions
        y_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save results
        with open('../results/evaluation_metrics.txt', 'w') as f:
            f.write(f"Selected Model: {best_name}\n")
            f.write(f"Mean Squared Error: {mse:.4f}\n")
            f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"R-squared Score: {r2:.4f}\n")
            f.write(f"\nCross-validation mean score: {best_score:.4f}\n")

        # Save feature coefficients
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': best_model.coef_
        }).sort_values('Coefficient', ascending=False)

        coefficients.to_csv('../results/model_coefficients.csv', index=False)
        print(f"\n✓ Model training complete. Best model: {best_name}")
        print("✓ Results saved to ../results/")

    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
