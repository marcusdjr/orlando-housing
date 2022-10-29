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
upstream = None

# This is a placeholder, leave it as None
product = None


# %%
from matplotlib.dviread import Page
import requests
from bs4 import BeautifulSoup
import pandas as pd
from csv import writer

url = "https://api.webscrapingapi.com/v1"
params = {
 "api_key": "WN5PAncn1XB2mzveYazxnJpsoIAq8KB8",
 "url": "https://www.realtor.com/realestateandhomes-search/Orlando_FL/radius-25"
}

response = requests.request("GET", url, params=params)

content = response.text

soup = BeautifulSoup(response.content, 'html.parser')
lists = soup.find_all('div', class_="jsx-11645185 summary-wrap")

with open('listing.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths','Sqft']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('span', attrs={'data-label': 'pc-price'}).text.replace('\n', '')
        beds = list.find('li', attrs={'data-label': 'pc-meta-beds'}).text.replace('\n', '')
        baths = list.find('li', attrs={'data-label': 'pc-meta-baths'}).text.replace('\n', '')
        sqft = list.find('li', attrs={'data-label': 'pc-meta-sqft'}).text.replace('\n', '')

        #info["Price"] = pd.to_numeric(info["price"], errors='coerce').fillna(0, downcast='infer')

        info = [price, beds, baths,sqft]
        thewriter.writerow(info)

# %%
