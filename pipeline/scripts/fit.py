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



# %%
#This file is scraped homes that we run the model againts- hom = homes on market
df = pd.read_csv(upstream['clean']['data'])
#Training csv
train = pd.read_csv('soldhomes.csv')
#Start of Analyzing/Training model

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
plt.figure(figsize=(10,10))
sns.jointplot(x=train.latitude, y=train.longitude, size=10)
plt.ylabel('Longitude',fontsize=12)
plt.ylabel('Laitude',fontsize=12)
plt.show()
sns.despine

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

y = train['price']

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
scaler = StandardScaler()

#y_train = scaler.fit_transform(pd.DataFrame(y_train))

#X_train = scaler.fit_transform(pd.DataFrame(X_train))

X_train = X_train[['listPrice']]

#Visualization that represents score
fig, ax = plt.subplots()
ax.scatter(y_train, X_train)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
#regression line
y_train, X_train['listPrice'] = y_train, X_train['listPrice']
ax.plot(y_train, LinearRegression().fit(y_train, X_train['listPrice']).predict(y_train))

plt.show()

# %%
#Score
from sklearn.preprocessing import StandardScaler

score = lm.score(X_train,y_train)

score


# %%
scaler = StandardScaler()

#Ys = scaler.fit_transform(pd.DataFrame(y_train))

#Ps = scaler.fit_transform(pd.DataFrame(X_train))

#y_predicted = predictions.values.reshape(1,-1)

#y_test = y_train.values.reshape(1,-1)

# %%
#Visualization that represents score
fig, ax = plt.subplots()
ax.scatter(y_train, X_train['price'])
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
regression line
y_train, X_train['price'] = y_train.reshape(-1,1), X_train['price'].reshape(-1,1)
ax.plot(y_train, LinearRegression().fit(y_train, X_train['price']).predict(y_train))

plt.show()

# %%
#Importing r2 score
from sklearn.metrics import r2_score

# %%
#Setting varialbles of Model 1
y_t = train['price']
x_t = train['listPrice']
model1 = (y_t,x_t)

# %%
#Score of Model 1, My model will give a more acurate prediction on how much the house will sell for than just going off of the listed price on Zillow.com
r_sqaured = r2_score(y_t,x_t)

# %%
r_sqaured

# %%
#Visualization that represents score of Model 1
fig, ax = plt.subplots()
ax.scatter(y_t, x_t)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
#regression line
y_t, x_t = y_t.values.reshape(-1,1), x_t.values.reshape(-1,1)
ax.plot(y_t, LinearRegression().fit(y_t, x_t).predict(y_t))

plt.show()

# %%
