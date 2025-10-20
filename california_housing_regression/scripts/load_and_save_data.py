from pathlib import Path
from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_and_save_data():
    """Load the sklearn California housing dataset and save as CSV.

    Paths are resolved relative to the project root (two levels up from this file).
    """
    print('Loading California Housing dataset...')
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.columns = ['Median_Income', 'House_Age', 'Average_Rooms', 'Average_Bedrooms',
                  'Population', 'Average_Occupancy', 'Latitude', 'Longitude', 'Median_House_Value']

    # Resolve paths relative to project root
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / 'california_housing.csv'

    df.to_csv(out_path, index=False)
    print(f'✅ Dataset saved to {out_path}')


if __name__ == '__main__':
    load_and_save_data()
