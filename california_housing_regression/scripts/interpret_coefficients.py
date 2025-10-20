from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base = Path(__file__).resolve().parent.parent
results_dir = base / 'results'
plots_dir = results_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

coefs_path = results_dir / 'model_coefficients.csv'
data_path = base / 'data' / 'california_housing.csv'

if not coefs_path.exists():
    raise SystemExit(f"model_coefficients.csv not found at {coefs_path}")
if not data_path.exists():
    raise SystemExit(
        f"data file not found at {data_path}. Run load_and_save_data.py first.")

coefs = pd.read_csv(coefs_path)
df = pd.read_csv(data_path)

# compute std for each feature
coefs['std'] = coefs['Feature'].map(df.std())
# per-SD effect in USD
coefs['effect_per_SD_usd'] = coefs['Coefficient'] * 100000
# per-unit effect in USD (Coefficient / std) * 100000
coefs['effect_per_unit_usd'] = (coefs['Coefficient'] / coefs['std']) * 100000

# round for readability
coefs[['Coefficient', 'std', 'effect_per_SD_usd', 'effect_per_unit_usd']] = coefs[[
    'Coefficient', 'std', 'effect_per_SD_usd', 'effect_per_unit_usd']].round(4)

# save interpreted table
out_table = results_dir / 'coefficients_interpreted.csv'
coefs.to_csv(out_table, index=False)

print('\nInterpreted coefficients saved to:', out_table)
print(coefs.to_string(index=False))

# Plot bar chart of effect_per_SD_usd (magnitude)
plt.figure(figsize=(10, 6))
sns.barplot(data=coefs.sort_values('effect_per_SD_usd', key=abs,
            ascending=False), x='effect_per_SD_usd', y='Feature', palette='vlag')
plt.xlabel('Effect per 1 SD (USD)')
plt.title('Coefficient effects (per 1 SD)')
plt.tight_layout()
plot_path = plots_dir / 'coefficients.png'
plt.savefig(plot_path)
print('\nPlot saved to:', plot_path)
