#ECA weather
# notes:
# - try to find humidity
# format output
# notes
# 
# Data from the European Climate Assessment & Dataset (ECA&D).
# Permision to use data granted as long as following source is acknowledged:
# ------------------------------------------------------------------------------#
# Klein Tank, A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface #
# air temperature and precipitation series for the European Climate Assessment. #
# Int. J. of Climatol., 22, 1441-1453.                                          #
# Data and metadata available at http://www.ecad.eu                             #
# ------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# data: https://www.ecad.eu/dailydata/predefinedseries.php#
file = 'C://Users//aidan//Documents//Physics//VScode//weather//ECA_london_weather_heathrow.csv'
df = pd.read_csv(file)
X = df.copy()

# Pre-processing ------------------------------------------------------------------------

#print(X.info())
# 'date' is int, everything else is float

#print(X.isnull().sum())
# a little less than 10% of the snow_depth data is missing

#(X['snow_depth']==0).sum()
# little less than 90% of the time, it doesnt snow

# it would be cool if i could impute NaN to zero in spirng, summer and autmumn
# and replace NaN with the non-zero mean in Winter. 
# would that be data-ethical?
# I think it would.

# Processing ----------------------------------------------------------------------------
# remove rows with missing target, set target
X.dropna(axis=0, subset=['sunshine'], inplace=True)
y = X.pop('sunshine')
#print(X.info())

# imputer
impute = SimpleImputer(strategy='mean')
X2 = pd.DataFrame(impute.fit_transform(X), columns=X.columns)
#print(X2.head())


# Identify High-Potential Features ------------------------------------------------------

# remove date column as I was having some problems processing it through MI
X2 = X2.drop(columns=['date'])
# for later
Features = X2.columns.tolist()

# Mutual Information function
def calc_MI(X, y):
    MI_scores = mutual_info_regression(X, y, discrete_features='auto', random_state=0)
    MI_scores = pd.Series(MI_scores, index=X.columns)
    MI_scores = MI_scores.sort_values(ascending=False)
    return MI_scores

MI_scores = calc_MI(X2, y)
print(MI_scores)

# Thoughts:

# The pressure seems really low, given that high/low pressure systems affect sunshine
# This could be because is summer, low pressure systems "trap" the lack of clouds above us
# Wheras in winter the same low pressure system will "trap" the clouds above us
# So maybe by differentiating the seasons, the model will be able to make use of the pressure data

# global radiation is really high potential 
#irradiance measurement in Watt per square meter (W/m2)
# cloud cover looks okay, but could definitely be higher since you know,
# thats whats blocking the sun..

# Create Lag Feature --------------------------------------------------------------------

# Use the previous days' sunshine to predict current sunshine
X2['sunshine_lag_1'] = y.shift(1)
X2['sunshine_lag_2'] = y.shift(2)
# drop first two rows of df since they will have NaN
X2.drop([0,1], inplace=True)
y = y.drop([0,1])
#print(X2.info())

# Create Season Feature -----------------------------------------------------------------

# The date will not be helpful to out predictive model, but maybe seasons/months will be.
X3 = X2.copy()
# Parse date
X3['date_parsed'] = pd.to_datetime(X['date'], format='%Y%m%d')

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

X3['Season'] = X3['date_parsed'].apply(get_season)

# Clean up columns and index
#X3 = X3.set_index('date')
X3 = X3.drop(['date_parsed'], axis=1)

'''
# Data Visualisation ---------------------------------------------------------------------

X3_with_target = X3.copy()
X3_with_target['sunshine'] = y
X3_with_target.info()

# Yesterday's Sunshine

f, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))
sns.regplot(data=X3_with_target,
           x='sunshine_lag_1',
           y='sunshine',
           scatter_kws={'s': 0.5},
           ax=axes[0]
           )
axes[0].set_title('Yesterday\'s Sunshine vs Sunshine')

sns.regplot(data=X3_with_target,
           x='sunshine_lag_2',
           y='sunshine',
           scatter_kws={'s': 0.5},
           ax=axes[1]
           )
axes[1].set_title('Day Before Yesterday\'s Sunshine vs Sunshine')
f.suptitle("Previous Day\s Sunshine vs Sunshine")
f.tight_layout()
plt.show()

# Evidently this feature will be less useful than I thought it would be haha

# Cloud Cover
sns.lmplot(data=X3_with_target, 
            x='cloud_cover',
            y ='sunshine',
            hue='Season',
            # change marker size
            scatter_kws={'s':2}
            )
plt.title("Cloud Cover vs Sunshine by Season")
plt.show()
# Two distinct trennds (Autumn+Winter) and (Spring+Summer)
# This will definitely help the model predict sunshine if it can differentiate seasons

# Global Radiation
g = sns.lmplot(data=X3_with_target, 
            x='global_radiation',
            y ='sunshine',
            col='Season',
            hue='Season',
            height=6,
            aspect=0.5,
            # change marker size
            scatter_kws={'s':2}
            )
g.set_axis_labels("Global Radiation (Wm^-2)", "Sunshine (Hrs)")
g.fig.suptitle("Global Radiation vs Sunshine by Season")
plt.tight_layout()
plt.show()
# Nice correlation when split by season. 

# Pressure

sns.histplot(data=X3_with_target,
             x='pressure',
             bins=30, 
             kde=True, 
             stat='count', 
             hue='Season', 
             label=' ')
plt.xlabel('Pressure (Pa)')
plt.ylabel('Counts')
plt.title('Distribution of Pressure measurements by Season')
plt.show()

sns.lmplot(data=X3_with_target, 
            x='pressure',
            y ='sunshine',
            col='Season',
            hue='Season',
            # change marker size
            scatter_kws={'s':2}
            )
plt.title('Pressure vs Sunshine by Season')
plt.show()

#Creat Log(pressure)
#X3_with_target['pressure2'] = X3_with_target.pressure.apply(np.exp)
#X3_with_target['pressure2'] = X3_with_target['pressure'] *10

sns.lmplot(data=X3_with_target, 
            x='pressure',
            y ='sunshine',
            col='Season',
            hue='Season',
            # change marker size
            scatter_kws={'s':2}
            )
plt.title('Pressure vs Sunshine by Season')
plt.show()


sns.displot(data=X3_with_target, 
            x="pressure",
            hue="Season", 
            )
plt.show()
'''
#  Feature Engineering -------------------------------------------------------------------

#model
# ordinal encoding bad bc implies an ordinal relationship between categories (i.e., that Winter < Spring < Summer < Fall), 
# which isn't appropriate for seasons as they don't have a natural ordering 

#target encoding:
# +Can capture some of the interaction between the category and the target variable.
# -can introduce target leakage if not handled carefully (e.g., through cross-validation)

# One hot because it does not impose any ordinal relationship between seasons
# One Hot Encode the seasons

one_hot = pd.get_dummies(X3['Season'])
X3 = X3.join(one_hot)
X3 = X3.drop(['Season'], axis=1)
#print(X3.info())

# The model wont be able to learn different relationships for each season independently if I dont have the interaction term.
# this way the model will learn relationships like "When the pressure is high during the summer, the sunshine is high" without learning that that is true for all seasons.

# Create Seasonal Interaction Features
# List of features we will "seasonalise" stored in "Features" from earlier
X4 = pd.DataFrame()
for feature in Features:
    X4[f'{feature}_winter'] = X3[feature] * X3['Winter']
    X4[f'{feature}_spring'] = X3[feature] * X3['Spring']
    X4[f'{feature}_summer'] = X3[feature] * X3['Summer']
    X4[f'{feature}_autumn'] = X3[feature] * X3['Autumn']

#print(X4.info())

# convert date_parsed to numerical features
#X3['year'] = X3['date_parsed'].dt.year
#X3['month'] = X3['date_parsed'].dt.month
#X3['day'] = X3['date_parsed'].dt.day


# perhaps the model will do better if rather than split by season,
# it is split using the K_means function?

# KMeans Clustering
X5 = X2.copy()
kmeans = KMeans(n_clusters=4, random_state=0)
X5['Cluster'] = kmeans.fit_predict(X5)
X5['Cluster'] = X5['Cluster'].astype('category')
X5 = pd.get_dummies(X5, columns=['Cluster'])

# Principle Component Analysis
X6 = X2.copy()

# Choose features to analyse with PCA
PCA_features = ['global_radiation', 'cloud_cover', 'mean_temp', 'pressure']
X6 = X6.loc[:, PCA_features]
# pressure component is low MI but I'm including it anyway to see if PCA will uncover
# its usefullness

# Standardise the data
standardise = StandardScaler()
X6 = standardise.fit_transform(X6)
mean = X6.mean(axis=0)
std = X6.std(axis=0)
print('mean:', mean)
print('std:', std)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X6)

# Convert to DF
component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns= component_names)

# MI
MI_scores = calc_MI(X_pca, y)
print(MI_scores)

X6 = X3.copy()
X6['PC1'] = X_pca['PC1']
print(X6.info())

# Model Building -------------------------------------------------------------------------

# Random Forest Model Function
def RFR_score(X, y):
    rfr_model = RandomForestRegressor(random_state=1)
    scores = -1 * cross_val_score(rfr_model,
                                  X,
                                  y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    mae = scores.mean()
    return mae

# XGBoost Regressor Function
def XGB_score(X,y):
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

    return mae

# Results DataFrame
mae_data = []
feature_list = ['Without Seasons', 
                'With Seasons', 
                'With Seasons * Features', 
                'With KMeans Clusters',
                'With Seasons and PCA']
# iterating through x and feature simultaneously to build df whilst calculating MAE
for x, feature in zip([X2, X3, X4, X5, X6], feature_list):
    forest_val = RFR_score(x, y)
    mae_data.append({'Feature': feature, 'Model': 'Random Forest', 'MAE': forest_val})
    print(f'The MAE for the Forest model {feature} is {forest_val:.3f}')

    xgb_val = XGB_score(x, y)
    mae_data.append({'Feature': feature, 'Model': 'XGBoost', 'MAE': xgb_val})
    print(f'The MAE for the XGBoost model {feature} is {xgb_val:.3f}')
    print('')

mae_df = pd.DataFrame(mae_data)

# Compare Results with a Barchart
plt.figure(figsize=(8,6))
sns.barplot(data= mae_df,
            x = 'Feature',
            y = 'MAE',
            hue = 'Model'
            )
plt.title('erm')
plt.xlabel('Feature')
plt.ylabel('MAE (+/- Hrs of Sunshine)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

                  
# Thoughts:
# Kmeans clustering scores almost identically to the model without any clustering
# It doesnt capture any of the relevent structures in the data, I've tried it with 
# lots of different n_clusters, it gets worse with more clusters. (Overfitting).
# Possible Futher Work:
# try a different clustering technique, like a gaussian model. I found some 'Gaussian
# Mixture Models' (GMM) online, this might do a better job.
