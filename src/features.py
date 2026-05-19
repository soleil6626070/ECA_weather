"""Feature engineering: seasons, lag features, seasonal interactions."""

import pandas as pd


# Create a Season column
def get_season(date):
    year = date.year

    # Spring equinox on March 20th
    if date >= pd.Timestamp(f'{year}-03-20') and date < pd.Timestamp(f'{year}-06-21'):
        return 'Spring'
    # Summer solstice June 21
    elif date >= pd.Timestamp(f'{year}-06-21') and date < pd.Timestamp(f'{year}-09-23'):
        return 'Summer'
    # Autumnal equinox September 23
    elif date >= pd.Timestamp(f'{year}-09-23') and date < pd.Timestamp(f'{year}-12-22'):
        return 'Autumn'
    # Winter solstice December 22
    else:
        return 'Winter'


# Create Lag Features
def add_lag_features(X, y):
    # Use the previous days sunshine to predict current sunshine
    X = X.copy()
    X['sunshine_lag_1'] = y.shift(1)
    X['sunshine_lag_2'] = y.shift(2)
    # drop first two rows of df since they will have NaN
    # use iloc not drop([0,1]): the index may not start at 0 after a previous dropna
    X = X.iloc[2:]
    y = y.iloc[2:]
    return X, y


# ~~~ One Hot Encoding ~~~
# ordinal encoding bad bc implies an ordinal relationship between categories
# (i.e., that Winter < Spring < Summer < Fall),
# which isn't appropriate for seasons as they don't have a natural ordering
# target encoding:
# +Can capture some of the interaction between the category and the target variable.
# -can introduce target leakage if not handled carefully (e.g., through cross-validation)
# One hot because it does not impose any ordinal relationship between seasons
def add_season(X):
    X = X.copy()
    X['Season'] = X['date'].apply(get_season)
    one_hot = pd.get_dummies(X['Season'])
    X = X.join(one_hot)
    X = X.drop(['Season'], axis=1)
    return X


# ~~~ Seasonal Interaction Features ~~~
# Perhaps the model wont be able to learn different relationships for each season independently
# if I dont have the interaction term.
# Perhaps this way the model will learn relationships like "When the pressure is high during the
# summer, the sunshine is high" without learning that that is true for all seasons.
def add_seasonal_interactions(X, features):
    # start from X.copy() not pd.DataFrame(): we want X augmented with the
    # interaction columns, not replaced by only the interaction columns
    X_new = X.copy()
    for feature in features:
        X_new[f'{feature}_winter'] = X[feature] * X['Winter']
        X_new[f'{feature}_spring'] = X[feature] * X['Spring']
        X_new[f'{feature}_summer'] = X[feature] * X['Summer']
        X_new[f'{feature}_autumn'] = X[feature] * X['Autumn']
    return X_new
