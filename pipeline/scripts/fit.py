# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = ['clean']

# This is a placeholder, leave it as None
product = None


# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics

import descartes
from shapely.geometry import Point, Polygon




# %%
#This file is scraped homes that we run the model againts- hom = homes on market
df = pd.read_csv(upstream['clean']['data'])
#Training csv
train = pd.read_csv('soldhomes.csv')
#Start of Analyzing/Training model
crs = {'init': 'epsg:4326'}

# %%
train.head()

# %%
#Most common bedroom number
train['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

# %%
#Visualizing the location of the houses based on latitude and longitude
import geopandas as gpd

florida = r'C:\Users\marcu\OneDrive\Desktop\Projects\florida shape file\tl_2019_12_place.shp'

florida_map = gpd.read_file(florida)

florida_map.plot()
geometry = [Point(xy) for xy in zip( train["longitude"], train["latitude"])]
geometry[:3]

geo_df = gpd.GeoDataFrame(train,
                          crs = crs,
                          geometry = geometry)
geo_df.head()

fig,ax = plt.subplots(figsize = (15,15))
florida_map.plot(ax, alpha = 0.4, color="grey")
geo_df{geo_df['WnvPresent'] == 0]
#plt.figure(figsize=(10,10))
#sns.jointplot(x=train.latitude, y=train.longitude, size=10)
#plt.ylabel('Longitude',fontsize=12)
#plt.xlabel('Laitude',fontsize=12)
#plt.show()
#sns.despine

#street_map = gpd.read_file('/Users/marcu/Downloads/tl_2019_12_place.zip')

# %%
#How Sqft is affecting the sale price of homes
plt.scatter(train.price,train.livingArea)
plt.title('Price vs Sqaure Feet')

# %%
#How Location is affecting the sale price of homes
plt.scatter(train.price,train.longitude)
plt.title('Price vs Location of the area')

# %%
#How Number of Bedrooms is affecting the sale price of homes
plt.scatter(train.bedrooms,train.price)
plt.title('Bedroom and Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()
sns.despine

# %%
#How the amount of days home was on zillow affected the sale price
plt.scatter(train.daysOnZillow,train.price)
plt.title('Days On Zillow vs Price')

# %%
#How the year the house was built affects the homes sale price
plt.scatter(train.yearBuilt,train.price)
plt.title('Year Built vs Price')

# %%
sns.distplot(train.price);

# %%
#Time to start training models

# %%
#Total Null Values
train.isnull().sum()

# %%
train.dtypes

# %%
train.head()

# %%
#Setting my X and y
X = train[['bedrooms','bathrooms','livingArea','listPrice']]
#'yearBuilt','bedrooms'

y = train[['price']]

# %%
#LASSO REGRESSION

#x = train.drop('price', axis=1).values
#y = train['price'].values
#features = train.drop('price',axis=1).columns
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)

lasso_coef = lasso.fit(X,y).coef_
plt.plot(range(len(features)),lasso_coef)

plt.xticks(range(len(features)),features,rotation=60)

plt.ylabel("Coefficients")

plt.show

# %%
#Train Test Split
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

# %%
#Linear Regression
from sklearn.linear_model import LinearRegression

# %%
lm = LinearRegression()

# %%
lm.fit(X_train,y_train)

# %%
#Predicting
predictions = lm.predict(X_train)
predictions

# %%
#Reshape
y_train, X_train = y_train.values.reshape(-1,1), X_train.values.reshape(-1,1)

#Visualization that represents score
fig, ax = plt.subplots()
ax.scatter(y_train, predictions)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
#regression line
y_train, predictions = y_train, predictions
ax.plot(y_train, LinearRegression().fit(y_train, predictions).predict(y_train))

plt.show()

# %%
#Importing Scores
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

# %%
#RMS calculation and value for "My Model"
rms = sqrt(mean_squared_error(y_train, predictions))
rms

# %%
#R2 calculation and value for "My Model"
r_sqaured = r2_score(predictions,y_train)

r_sqaured


# %%
#RMSLE calculation and value for "My Model"
mean_squared_log_error(predictions,y_train, squared=False)


# %%
#MAE calculation and value for "My Model"
def mae(predictions,y_train):
    predictions,y_train = np.array(y_train), np.array(predictions)
    return np.mean(np.abs(predictions - y_train))

print(mae(predictions,y_train))

# %%
#lasso = Lasso(alpha=1.0)

# %%
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(lasso, y_train, predictions, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#scores = absolute(scores)
# force scores to be positive
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#ENR = ElasticNet(alpha=1.0, l1_ratio=0.5)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(ENR, y_train, predictions, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#Linear Regression Model(List Price)
#Setting varialbles of Model 1
y_t = train['price']
x_t = train['listPrice']
model1 = (y_t,x_t)

# %%
#RMS calculation and value for "Linear Regression Model(List Price)"
rms1 = sqrt(mean_squared_error(y_t,x_t))

# %%
rms1

# %%
#R Sqaure Score of Linear Regression Model(List Price), My model will give a more acurate prediction on how much the house will sell for than just going off of the listed price on Zillow.com
r_sqaured1 = r2_score(y_t,x_t)

# %%
r_sqaured1

# %%
#RMSLE calculation and value for Linear Regression Model(List Price)
mean_squared_log_error(y_t,x_t, squared=False)


# %%
#MAE calculation and value for Linear Regression Model(List Price)
def mae(x_t,y_t):
    x_t,y_t = np.array(y_t), np.array(x_t)
    return np.mean(np.abs(x_t - y_t))

print(mae(x_t,y_t))

# %%
#Reshape
#y_t,x_t = y_t.values.reshape(-1,1), x_t.values.reshape(-1,1)
#lASSO calculation and value for Linear Regression Model(List Price)
#lasso = Lasso(alpha=1.0)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(lasso, y_t,x_t, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#scores = absolute(scores)
# force scores to be positive
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#ENR calculation and value for Linear Regression Model(List Price,Sqft)
#ENR = ElasticNet(alpha=1.0, l1_ratio=0.5)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(ENR, y_t,x_t, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#Visualization that represents score of Linear Regression Model(List Price)
fig, ax = plt.subplots()
ax.scatter(y_t, x_t)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
#regression line
#y_t, x_t = y_t.values.reshape(-1,1), x_t.values.reshape(-1,1)
ax.plot(y_t, LinearRegression().fit(y_t, x_t).predict(y_t))

plt.show()

# %%
#Linear Regression Model(List Price,Sqft)
#Setting my X and y
X = train[['livingArea','listPrice']]
#'yearBuilt','bedrooms'

y = train[['price']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

# %%
lm.fit(X_train,y_train)

# %%
#Predicting
predictions = lm.predict(X_train)
predictions

# %%
#Reshape
y_train, X_train = y_train.values.reshape(-1,1), X_train.values.reshape(-1,1)

#Visualization that represents score
fig, ax = plt.subplots()
ax.scatter(y_train, predictions)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
#regression line
y_train, predictions = y_train, predictions
ax.plot(y_train, LinearRegression().fit(y_train, predictions).predict(y_train))

plt.show()

# %%
# R Sqaure Score of Linear Regression Model(List Price,Sqft)
r_sqaured = r2_score(predictions,y_train)

r_sqaured

# %%
#RMS Score of Linear Regression Model(List Price,Sqft)
rms = sqrt(mean_squared_error(y_train, predictions))
rms

# %%
#RMSLE calculation and value for Linear Regression Model(List Price,Sqft)
mean_squared_log_error(y_train, predictions, squared=False)


# %%
#MAE calculation and value for Linear Regression Model(List Price,Sqft)
def mae(predictions,y_train):
    predictions,y_train = np.array(y_train), np.array(predictions)
    return np.mean(np.abs(predictions - y_train))

print(mae(predictions,y_train))

# %%
#lASSO calculation and value for Linear Regression Model(List Price,Sqft)
#lasso = Lasso(alpha=1.0)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(lasso, predictions,y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#scores = absolute(scores)
# force scores to be positive
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#ENR calculation and value for Linear Regression Model(List Price,Sqft)
#ENR = ElasticNet(alpha=1.0, l1_ratio=0.5)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(ENR, y_train, predictions, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#Linear Regression Model(List Price,Sqft,Zipcode)

# %%
#Linear Regression Model(List Price,Sqft)
#Setting my X and y
X = train[['livingArea','listPrice','address_zipcode']]

y = train[['price']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

# %%
lm.fit(X_train,y_train)

# %%
#Predicting
predictions = lm.predict(X_train)
predictions

# %%
#Reshape
y_train, X_train = y_train.values.reshape(-1,1), X_train.values.reshape(-1,1)

#Visualization that represents score
fig, ax = plt.subplots()
ax.scatter(y_train, predictions)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
#regression line
y_train, predictions = y_train, predictions
ax.plot(y_train, LinearRegression().fit(y_train, predictions).predict(y_train))

plt.show()

# %%
# R Sqaure Score of Linear Regression Model(List Price,Sqft,Zipcode)
r_sqaured = r2_score(predictions,y_train)

r_sqaured

# %%
#RMS Score of Linear Regression Model(List Price,Sqft,Zipcode)
rms = sqrt(mean_squared_error(y_train, predictions))
rms

# %%
#RMSLE calculation and value for Linear Regression Model(List Price,Sqft,Zipcode)
mean_squared_log_error(y_train, predictions, squared=False)


# %%
#MAE calculation and value for Linear Regression Model(List Price,Sqft,Zipcode)
def mae(predictions,y_train):
    predictions,y_train = np.array(y_train), np.array(predictions)
    return np.mean(np.abs(predictions - y_train))

print(mae(predictions,y_train))

# %%
#Lasso calculation and value for Linear Regression Model(List Price,Sqft,Zipcode)
#lasso = Lasso(alpha=1.0)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(lasso, predictions,y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#scores = absolute(scores)
# force scores to be positive
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#ENR calculation and value for Linear Regression Model(List Price,Sqft,Zipcode)
#ENR = ElasticNet(alpha=1.0, l1_ratio=0.5)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(ENR, y_train, predictions, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
#GBC calculation and value for Linear Regression Model(List Price,Sqft,Zipcode)
#GBC = GradientBoostingClassifier()
#GBC.fit(y_train, predictions)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(GBC, y_train, predictions, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# %%
