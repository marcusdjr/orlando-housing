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
lists = soup.find_all('li', class_="component_property-card")

with open('listing.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('a', class_="listing-price").text.replace('\n', '')
        beds = list.find('div', class_="property-beds").text.replace('\n', '')
        baths = list.find('div', class_="property-baths").text.replace('\n', '')
        #sqft = list.find('div', class_="property-sqft").text.replace('\n', '')
        #sqft = list.find('div', class_="property-sqft").text.replace('\n', '')
        
        info = [price, beds, baths]
        thewriter.writerow(info)