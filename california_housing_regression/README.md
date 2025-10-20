# California Housing Price Prediction

This project predicts median house values in California using Linear Regression.

## Folder Structure
- data/: dataset storage
- scripts/: Python scripts for loading, training, and visualization
- results/: model outputs and plots
- report/: project documentation

## How to Run
There are two ways to run the project:

Option A — single command (recommended)

From the repository root run the helper PowerShell script `run_all.ps1` which will create a virtual environment, install dependencies, create the dataset, and train the model:

```powershell
# run from repository root
.\run_all.ps1
```

Option B — run scripts manually

1. Run `scripts/load_and_save_data.py`
2. Run `scripts/train_linear_regression.py`
3. Check metrics and plots in `results/`

## Model evaluation

After running the training script the evaluation metrics are written to `results/evaluation_metrics.txt`. The latest results are:

- Mean Squared Error (MSE): 0.4388
- Root Mean Squared Error (RMSE): 0.6624
- Mean Absolute Error (MAE): 0.4823
- R-squared: 0.6682
- Cross-validation mean score (CV R²): 0.6742

What these values mean (plain English):
- MSE (0.4388) is the average of the squared differences between predicted and actual values; lower is better but it's in squared target units so it's less intuitive on its own.
- RMSE (0.6624) is the square root of MSE and represents the typical prediction error on the same scale as the target. For the California Housing dataset the target is the median house value in units of 100,000 USD, so an RMSE of 0.6624 corresponds to about $66,240.
- MAE (0.4823) is the average absolute prediction error (less sensitive to large outliers). An MAE of 0.4823 corresponds to about $48,230.
- R-squared (0.6682) means the model explains about 66.8% of the variance in the target variable. This indicates a reasonably strong linear relationship but leaves substantial unexplained variation.
- Cross-validation mean score (0.6742) gives an estimate of the model's expected R² on unseen folds; it aligns with the single-run R² and suggests the model generalizes similarly across folds.

Implications and next steps:
- The model performs reasonably well for a simple linear regression: it captures much of the signal (R² ≈ 0.67) but typical prediction errors are still on the order of tens of thousands of dollars.
- To reduce error and improve generalization consider:
	- Feature engineering: create interaction terms, polynomial features, or spatial aggregations (neighborhood-level features).
	- Try regularized or non-linear models (Ridge/Lasso, Random Forest, Gradient Boosting, or XGBoost).
	- Transform the target (e.g., log transform) if the error distribution is skewed.
	- Add more predictive features (external data) or use geographic clustering.
	- Evaluate residual plots to check heteroskedasticity or systematic bias.

These metrics and suggestions are also saved in `results/` so you can iterate and compare future runs.

Install dependencies with:
```
pip install -r requirements.txt
```

## Model coefficients interpretation

The trained linear model coefficients are saved in `results/model_coefficients.csv`. Below is an interpretation of the most recent coefficients (coefficients are on the target scale of median house value in units of 100,000 USD):

- Median_Income: 0.8697 — A one standard-deviation increase in median income is associated with about +$86,970 in median house value. Per one unit of Median_Income (original units) this corresponds to roughly +$45,778.
- Average_Bedrooms: 0.1788 — One SD increase corresponds to about +$17,883; per one bedroom (original units) ≈ +$37,734.
- House_Age: 0.1506 — One SD increase corresponds to about +$15,058; per one year of house age ≈ +$1,196 (age has a large SD so per-year effect is small).
- Population: 0.0459 — One SD increase corresponds to about +$4,591; per one person increase ≈ +$4.05 (negligible per-person effect, more relevant at neighborhood scale).
- Average_Rooms: -0.1891 — One SD increase corresponds to about -$18,908; per one room increase ≈ -$7,642.
- Average_Occupancy: -0.2573 — One SD increase corresponds to about -$25,726; per one occupant increase ≈ -$2,477.
- Longitude: -0.7952 — One SD increase corresponds to about -$79,515; longitude's sign indicates east/west location correlates strongly with price in this dataset. Per degree longitude ≈ -$39,688.
- Latitude: -0.8602 — One SD increase corresponds to about -$86,021; latitude is similarly strongly associated; per degree latitude ≈ -$40,273.

Caveats and notes:
- These are linear, ceteris-paribus associations: each coefficient estimates the effect of that feature holding the other features constant. They are not necessarily causal.
- Some features (like latitude/longitude) capture location effects; interpreting them as "per degree" is coarse — spatial effects are typically non-linear and better captured with location-specific features or spatial models.
- The model used standardized features during training. The per-unit dollar translations above were computed by combining coefficients with feature standard deviations so they are approximate and intended for intuition.
- Multicollinearity or omitted variables can distort coefficient magnitudes; consider checking feature correlations and variance inflation factors (VIF) if you plan to interpret coefficients carefully.
