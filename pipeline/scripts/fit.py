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
#missing_values=['??','na','X','999999','NaN']
#X=X.replace(missing_values,np.NaN)
#X

#m=round(df["bedrooms"].mean(),2)
#m
#X["bedrooms"].fillna(m,inplace=True)


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
train.isnull().sum()

# %%
missing_values=['??','na','X','999999']
train=train.replace(missing_values,np.NaN)

# %%
train.dtypes

# %%
#Filling in missing Price rows
train["price"]=train["price"].astype("float64")
m=round(train["price"].mean(),2)

train["price"].fillna(m,inplace=True)

#Filling in missing Bedroom rows
train["bedrooms"]=train["bedrooms"].astype("float64")
m=round(train["bedrooms"].mean(),2)

train["bedrooms"].fillna(m,inplace=True)

#Filling in missing Bathroom rows
train["bathrooms"]=train["bathrooms"].astype("float64")
m=round(train["bathrooms"].mean(),2)

train["bathrooms"].fillna(m,inplace=True)

#Filling in missing yearBuilt rows
train["yearBuilt"]=train["yearBuilt"].astype("float64")
m=round(train["yearBuilt"].mean(),2)

train["yearBuilt"].fillna(m,inplace=True)

#Filling in missing livingArea  rows
train["livingArea "]=train["livingArea"].astype("float64")
m=round(train["livingArea"].mean(),2)

train["livingArea"].fillna(m,inplace=True)

#Filling in missing livingArea  rows
train["daysOnZillow"]=train["daysOnZillow"].astype("float64")
m=round(train["daysOnZillow"].mean(),2)

train["daysOnZillow"].fillna(m,inplace=True)

# %%
X = train[['bedrooms', 'bedrooms','bathrooms', 'yearBuilt','livingArea','price']]

y = train['price']

# %%
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
predictions = lm.predict(X_test)
predictions

# %%
lm.score(X_train,y_train)

# %%
#KNN 

from sklearn.model_selection import train_test_split

X = train[['bedrooms', 'bedrooms','bathrooms', 'yearBuilt','livingArea','price']]

y = train['price']

from sklearn.neighbors import KNeighborsClassifier

# %%
knn = KNeighborsClassifier(n_neighbors=1)

# %%
knn.fit(X_train,y_train)

# %%
pred = knn.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix

# %%
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# %%
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# %%
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# %%
