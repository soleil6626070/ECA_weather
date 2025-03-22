# This code will:
# 1) Read in a data file downloaded from ECAD
# 2) Clean the data
# 3) Visualise the data
# 4) Experiment with different feature to see how machine
#    learning models react to them.
# 5) Build, train, test a Random Forest Regresson model & 
#    a Gradient Boost model.
# 6) Predict the daily hours of sunshine at London Heathrow

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

#print((X['snow_depth']==0).sum())
# little less than 90% of the time, it doesnt snow

# I could impute missing values of snow_depth to zero in spirng, summer and autmumn
# and replace NaN with the previous days value in winter.


# Processing ----------------------------------------------------------------------------

# remove rows with missing target
X.dropna(axis=0, subset=['sunshine'], inplace=True)
# set target
y = X.pop('sunshine')
#print(X.info())

# Snow depth imputing -------------------------------------------------------------------
X['date_parsed'] = pd.to_datetime(X['date'], format='%Y%m%d')

month = X.date_parsed.dt.month
day = X.date_parsed.dt.day

# Backfill winter months, fill to zero the rest of the year
winter_mask = ((month==12) & (day >= 21) | (month==1) | (month==2) | (month==3) & (day <= 20))
X.loc[winter_mask, 'snow_depth'] = X.loc[winter_mask, 'snow_depth'].bfill(axis=0, limit=1).fillna(0)
X.loc[~winter_mask, 'snow_depth'] = X.loc[~winter_mask, 'snow_depth'].fillna(0)

#print(X.isnull().sum())    snow_depth missing values successfully replaced
X.drop(columns=['date_parsed'], inplace=True)

# Imputer -------------------------------------------------------------------------------
impute = SimpleImputer(strategy='mean')
X2 = pd.DataFrame(impute.fit_transform(X), columns=X.columns)
#print(X2.head())
#print(X2.isnull().sum())

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

# global radiation is really high potential, would be perfect for a linear regression model.
# Irradiance measurement in Watt per square meter (W/m2)
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

# The date will not be helpful to our predictive model, but maybe seasons/months will be.
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


# Data Visualisation ---------------------------------------------------------------------

# Uncomment the < ''' > pairs for the graphs you would like to see

X3_with_target = X3.copy()
X3_with_target['sunshine'] = y
X3_with_target.info()

# Yesterday's Sunshine
#'''
# Correlation Calc using the .corr() function
corr1 = X3_with_target['sunshine'].corr(X3_with_target['sunshine_lag_1']).round(3)
corr2 = X3_with_target['sunshine'].corr(X3_with_target['sunshine_lag_2']).round(3)

f, axes = plt.subplots(1, 2, figsize=(10, 7), sharex=True, sharey=True) #gridspec_kw=dict(width_ratios=[4, 3]))
sns.regplot(data=X3_with_target,
           x='sunshine_lag_1',
           y='sunshine',
           scatter_kws={'s': 1.0, 'alpha': 0.5},
           ax=axes[0]
           )
axes[0].set_title('Yesterday\'s Sunshine vs Sunshine')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.6, linestyle='--')
axes[0].set_xlabel('Yesterday\'s Sunshine (Hrs)')
axes[0].set_ylabel('Sunshine (Hrs)')
axes[0].text(14, 7, f"r = {corr1}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

sns.regplot(data=X3_with_target,
           x='sunshine_lag_2',
           y='sunshine',
           scatter_kws={'s': 1.0, 'alpha': 0.5},
           ax=axes[1]
           )
axes[1].set_title('Day Before Yesterday\'s Sunshine vs Sunshine')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.6, linestyle='--')
axes[1].set_xlabel('Day Before Yesterday\'s Sunshine (Hrs)')
axes[1].set_ylabel('Sunshine (Hrs)')
axes[1].text(14, 6, f"r = {corr2}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

f.suptitle("Previous Days Sunshine vs Sunshine", fontsize=16, fontstyle="oblique")
f.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.5, 0.1, "Slightly better garbage on the left than the right. This feature will be less useful than I thought.", 
           ha="center", fontsize=12, fontstyle="italic")
plt.gcf().set_facecolor('#D3D3D3')
plt.show()

#'''

# Cloud Cover
#'''
g = sns.FacetGrid(
    X3_with_target, 
    col="Season",
    col_wrap=2,
    height=5,
    aspect=1.2
)

g.map_dataframe(sns.boxenplot,  
    x = 'cloud_cover',
    y = 'sunshine',
    palette = 'pastel',
    )

g.set_titles("{col_name}", fontsize=16)
g.set_axis_labels("Cloud Cover (Hrs)", "Sunshine (Hrs)", fontsize=16)
g.fig.suptitle("Cloud Cover vs Sunshine by Season", fontsize=18)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.5, 0.05, "These plots are great as they identified data corruption in the dataset that I had missed. \n"
                    "There are 19 rows in the dataset with the same abnormal value for cloud cover. \n"
                    "These could be floating point precision errors.", 
                    ha="center", fontsize=16, fontstyle="italic", wrap=True)
plt.gcf().set_facecolor('#D3D3D3')
plt.show()

# Finding anomalies
x3_corrupted = X3_with_target[(X3_with_target['cloud_cover'] > 5.0) & (X3_with_target['cloud_cover'] < 6.0)]
#print(x3_corrupted[['cloud_cover']].shape)
# I now experience the pain of not having a pipeline

#'''

# Global Radiation
#'''
g = sns.lmplot(data=X3_with_target, 
            x='global_radiation',
            y ='sunshine',
            col='Season',
            hue='Season',
            height=6,
            aspect=0.5,
            palette = 'colorblind',
            # change marker size
            scatter_kws={'s':2}
            )
g.set_axis_labels("Global Radiation (Wm^-2)", "Sunshine (Hrs)")
g.fig.suptitle("Global Radiation vs Sunshine by Season")
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.figtext(0.06, 0.1, " Two distinct patterns for Spring & Summer vs Autumn & Winter.\n"
" Also highlight some artifacts in the Spring & Summer seasons where the sunshine is between 12-15 hours"
" with global radiation at only two quantised values.", 
ha="left", fontsize=12, fontstyle="italic", wrap=True)
plt.gcf().set_facecolor('#D3D3D3')
plt.show()
#'''

# Pressure lmplots
'''
# awful plot gives us nothing, lets see if a logarithm improves it
sns.lmplot(data=X3_with_target, 
            x='pressure',
            y ='sunshine',
            col='Season',
            hue='Season',
            palette='colorblind',
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

# Nope thats just as terrible. Lets try a histogram
'''

# Pressure histogram
#'''
sns.histplot(data=X3_with_target,
             x='pressure',
             bins=30, 
             kde=True, 
             stat='count', 
             hue='Season',
             palette='bright',
             label=' ')
plt.xlabel('Pressure (Pa)')
plt.ylabel('Counts')
plt.title('Distribution of Pressure measurements by Season')
plt.subplots_adjust(bottom=0.3)
plt.figtext(0.5, 0.1, "Almost exemplary collection of Gaussian distributions\n"
" Since our data is normally distributed, our dataset is likely not too small. Reassuring.",
ha="center", fontsize=12, fontstyle="italic", wrap=True)
plt.show()
#Beautiful
#'''

# Pressure Displot
'''
# Displot for fun. I prefer the previous histogram though
sns.displot(data=X3_with_target, 
            x="pressure",
            hue="Season",
            palette='bright' 
            )
plt.show()
'''

#  Feature Engineering -------------------------------------------------------------------

# ordinal encoding bad bc implies an ordinal relationship between categories 
# (i.e., that Winter < Spring < Summer < Fall), 
# which isn't appropriate for seasons as they don't have a natural ordering 
#target encoding:
# +Can capture some of the interaction between the category and the target variable.
# -can introduce target leakage if not handled carefully (e.g., through cross-validation)
# One hot because it does not impose any ordinal relationship between seasons

# ~~~ One Hot Encoding ~~~

one_hot = pd.get_dummies(X3['Season'])
X3 = X3.join(one_hot)
X3 = X3.drop(['Season'], axis=1)
#print(X3.info())

# Perhaps the model wont be able to learn different relationships for each season independently 
# if I dont have the interaction term.
# Perhaps this way the model will learn relationships like "When the pressure is high during the 
# summer, the sunshine is high" without learning that that is true for all seasons.

# ~~~ Seasonal Interaction Features ~~~
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

# ~~~ KMeans Clustering ~~~
# The kmeans is sensitive to to scale, scaled features perform better. 
# However we should absolutely not scale features that are directly comparable, like mean temp and max temp.
X5 = X2.copy()
kmeans_features = list( set(Features) - set(['mean_temp', 'min_temp']) )
X5_scaled = X5.loc[:, kmeans_features]
X5_scaled = (X5_scaled - X5_scaled.mean(axis=0)) / X5_scaled.std(axis=0)

kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
X5['Cluster'] = kmeans.fit_predict(X5_scaled)
X5['Cluster'] = X5['Cluster'].astype('category')
X5 = pd.get_dummies(X5, columns=['Cluster'])

# ~~~ Principle Component Analysis ~~~
X6 = X2.copy()

# Choose features to analyse with PCA
PCA_features = ['global_radiation', 'cloud_cover', 'mean_temp', 'pressure']
X6 = X6.loc[:, PCA_features]
# pressure component is low MI but I'm including it anyway to see if PCA will uncover
# its usefullness

# Standardise the data (different scaling method for fun)
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

# Make df without seasons + PCA
X6 = X2.copy()
X6['PC1'] = X_pca['PC1']
print(X6.info())

# Make df with seasons + PCA
X7 = X3.copy()
X7['PC1'] = X_pca['PC1']

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

    # Getting feature importance
    # Have to refit as im using cross validate
    rfr_model.fit(X, y)
    rfr_FI = rfr_model.feature_importances_
    return mae, rfr_FI

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

    # Getting feature importance
    xgbr_FI = xgbr_model.feature_importances_
    return mae, xgbr_FI

# Results DataFrame
mae_data = []
feature_list = ['Without Seasons', 
                'With Seasons', 
                'With Seasons * Features', 
                'With KMeans Clusters',
                'With PCA but Without Seasons',
                'With PCA and Seasons']
# iterating through x and feature simultaneously to build df whilst calculating MAE
for x, feature in zip([X2, X3, X4, X5, X6, X7], feature_list):
    forest_val, rfr_FI = RFR_score(x, y)
    mae_data.append({'Feature': feature, 'Model': 'Random Forest', 'MAE': forest_val})
    print(f'The MAE for the Forest model {feature} is {forest_val:.3f}')
    # uncomment for feature importance
    #print('with feature importances: ', rfr_FI)
    print('')

    xgb_val, xgbr_FI = XGB_score(x, y)
    mae_data.append({'Feature': feature, 'Model': 'XGBoost', 'MAE': xgb_val})
    print(f'The MAE for the XGBoost model {feature} is {xgb_val:.3f}')
    # uncomment for feature importance
    #print('with feature importances: ', xgbr_FI)
    print('')
    print('')


mae_df = pd.DataFrame(mae_data)

# Compare Results with a Barchart
plt.figure(figsize=(8,6))
plt.gcf().set_facecolor('#f0f0f0')
ax = sns.barplot(data= mae_df,
                 x = 'Feature',
                 y = 'MAE',
                 hue = 'Model'
                 )

plt.title('Comparing the Mean Absolute Error of the XGBoost and\nRandomForestRegressor models for datasets with differing features.', fontsize=14)
plt.xlabel('Feature')
plt.ylabel('MAE (+/- Hrs of Sunshine)')
plt.legend(loc='lower left')
plt.text(0.95, 0.05, 'Lower is better!', 
         horizontalalignment='right', 
         verticalalignment='bottom', 
         transform=plt.gca().transAxes, 
         fontsize=14, 
         fontstyle='italic', 
         color='gray',
         # bbox to add the backround box behind the text
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='lightgray', boxstyle='round'))
plt.xticks(rotation=45)
# value labels, have to iterate through both types of bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='edge', padding=-0.5, color='black', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Find the best model
best_model = mae_df.loc[mae_df['MAE'].idxmin()]
# Output
print('The', best_model['Model'], best_model['Feature'], 'is able to predict the')
print('daily sunshine at London Heathrow with a mean absolute error of ')
print(f'{(best_model['MAE']*60):.2g} minutes.')

# Thoughts:
#
# The gradient boost model outperformed the random forest model on every occasion.
#
# The season interaction features: X4[f'{feature}_season'] = X3[feature] * X3['Season']
# were useless, the hypothesis of it helping was incorrect.
#
# Kmeans clustering scores almost identically to the model without any clustering
# It doesnt capture any of the relevent structures in the data, I've tried it with 
# lots of different n_clusters, it gets worse with more clusters. (Overfitting).
# I've now output the feature importance, we can see that for the RFR model the
# cluster features are unimportant, however for the XGBR cluster_0 is actually
# rather important at 0.162. Interesting.

# Possible Futher Work:
# try a different clustering technique, like a gaussian model. I found some 'Gaussian
# Mixture Models' (GMM) online, this might do a better job.
