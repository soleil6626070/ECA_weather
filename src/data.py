"""Loading and cleaning the ECA&D London Heathrow daily weather dataset."""

from pathlib import Path

import pandas as pd

# data: https://www.ecad.eu/dailydata/predefinedseries.php#
# Path is anchored to this file so load_raw() works from anywhere
DEFAULT_CSV = Path(__file__).parent.parent / 'data' / 'ECA_london_weather_heathrow.csv'


def load_raw(path=DEFAULT_CSV):
    df = pd.read_csv(path)
    # parse the YYYYMMDD int once, here, so downstream code can use df['date'].dt
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df


# Snow depth imputing
# I could impute missing values of snow_depth to zero in spring, summer and autumn
# and replace NaN with the previous days value in winter.
def clean(df):
    # remove rows with missing target
    df = df.dropna(axis=0, subset=['sunshine']).copy()

    month = df.date.dt.month
    day = df.date.dt.day

    # Backfill winter months, fill to zero the rest of the year.
    # Boundaries match get_season(): Winter = Dec 22 - Mar 19 (astronomical solstice/equinox)
    winter_mask = ((month==12) & (day >= 22) | (month==1) | (month==2) | (month==3) & (day <= 19))
    df.loc[winter_mask, 'snow_depth'] = df.loc[winter_mask, 'snow_depth'].bfill(axis=0, limit=1).fillna(0)
    df.loc[~winter_mask, 'snow_depth'] = df.loc[~winter_mask, 'snow_depth'].fillna(0)

    return df
