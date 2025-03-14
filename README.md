This code uses daily weather data at London Heathrow to build and compare a linear regression model against a gradient boosting model to predict the daily hours of sunshine. 

This code can be broken down into different sections:
1. Cleaning the data and imputing missing values.
2. Visualising the data with the Matplotlib and Seaborn libraries.
3. Creating new features to see how the machine learning models would react to them.
4. Building the ML models, applying them to the tweaked datasets & finally comparing them.

My closing thoughts on the project:

A) Realistically, there are not many uses for this code; the only one I can think of is if the sunshine sensor was malfunctioning that day and you wanted to impute the missing value for that day with a prediction rather than have a missing value for the sake of the record book.

B) However I still consider this to be a great success, as I have familiarised myself with the Scikit-Learn library, gained deeper insight into what wrangling techniques work for the specified models and more importantly which ones don't. Plus I had fun creating pretty plots wich I hadn't done in a while.

C) Overall I'd say my learning goals were accomplished and I can move on satisfied.

The data was downloaded from: https://www.ecad.eu/dailydata/predefinedseries.php#

Permision to use data granted as long as following source is acknowledged:
# -------------------------------------------------------------------------
# Klein Tank, A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface 
# air temperature and precipitation series for the European Climate Assessment. 
# Int. J. of Climatol., 22, 1441-1453.                                          
# Data and metadata available at http://www.ecad.eu                             
# -------------------------------------------------------------------------
