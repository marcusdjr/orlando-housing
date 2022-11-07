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
hom = pd.read_csv(upstream['clean']['data'])
#Start of Analyzing/Training model

# %%
df.head()

# %%
#Most common bedroom number
df['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

# %%
#Visualizing the location of the houses based on latitude and longitude
plt.figure(figsize=(10,10))
sns.jointplot(x=df.latitude, y=df.longitude, size=10)
plt.ylabel('Longitude',fontsize=12)
plt.ylabel('Laitude',fontsize=12)
plt.show()
sns.despine

# %%
#How Sqft is affecting the sale price of homes
plt.scatter(df.price,df.livingArea)
plt.title('Price vs Sqaure Feet')

# %%
#How Location is affecting the sale price of homes
plt.scatter(df.price,df.longitude)
plt.title('Price vs Location of the area')

# %%
#How Number of Bedrooms is affecting the sale price of homes
plt.scatter(df.bedrooms,df.price)
plt.title('Bedroom and Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()
sns.despine

# %%
#How the amount of days home was on zillow affected the sale price
plt.scatter(df.daysOnZillow,df.price)
plt.title('Days On Zillow vs Price')

# %%
#How the year the house was built affects the homes sale price
plt.scatter(df.yearBuilt,df.price)
plt.title('Year Built vs Price')

# %%
#Time to start training models

# %%
df.info()

# %%
from sklearn.model_selection import train_test_split

# %%
X = df[['bedrooms', 'bathrooms', 'yearBuilt',
       'livingArea']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# %%
X.isnull().sum()

# %%
missing_values=['??','na','X','999999','NaN']
X=X.replace(missing_values,np.NaN)
X

# %%
m=round(df["bedrooms"].mean(),2)
m
X["bedrooms"].fillna(m,inplace=True)
X["bathrooms"].fillna(m,inplace=True)
X["yearBuilt"].fillna(m,inplace=True)
X["livingArea"].fillna(m,inplace=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc = RandomForestClassifier(n_estimators=200)

# %%
rfc.fit(X_train,y_train)

# %%
rfc_pred = rfc.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix

# %%
print (confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

# %%
rfc_pred

# %%
