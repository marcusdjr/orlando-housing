from matplotlib.dviread import Page
import requests
from bs4 import BeautifulSoup
import pandas as pd
from csv import writer

url = "https://api.webscrapingapi.com/v1"
params = {
 "api_key": "WN5PAncn1XB2mzveYazxnJpsoIAq8KB8",
 "url": "https://www.realtor.com/realestateandhomes-search/Orlando_FL"
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