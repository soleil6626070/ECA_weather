"""Model evaluation helpers.

RFR_score and XGB_score are the original functions from the legacy first attempt.
They contain the temporal leakage bugs. These are to be fixed with cv_score
with TimeSeriesSplit. Keeping them to compare with fix.
"""

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# Random Forest Model Function
def RFR_score(X, y):
    rfr_model = RandomForestRegressor(random_state=1)
    scores = -1 * cross_val_score(rfr_model,
                                  X,
                                  y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    mae = scores.mean()

    # Getting feature importance
    # Have to refit as im using cross validate
    rfr_model.fit(X, y)
    rfr_FI = rfr_model.feature_importances_
    return mae, rfr_FI


# XGBoost Regressor Function
def XGB_score(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    xgbr_model = XGBRegressor(random_state=0,
                              n_estimators=1000,
                              early_stopping_rounds=5,
                              learning_rate=0.1,
                              n_jobs=4)

    xgbr_model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      verbose=False)

    y_prediction = xgbr_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_prediction)

    # Getting feature importance
    xgbr_FI = xgbr_model.feature_importances_
    return mae, xgbr_FI