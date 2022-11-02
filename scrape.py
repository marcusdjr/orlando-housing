from bs4 import BeautifulSoup
import requests
from requests_html import HTML, HTMLSession
from csv import writer
from urllib.request import urlopen
import pandas as pd
from random import randint
from time import sleep


url= "https://www.century21.com/real-estate/orlando-fl/LCFLORLANDO/"

for page in range(1,10):

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    lists = soup.find_all('div', class_="property-card-primary-info")

    for i in range(4,19):
        if page>1:
            print(f"{(i-3)+page*15}" + lists[i].text)
        else:
            print(f"{i-3}" + lists[i].text)
  
    sleep(randint(2,10))

with open('test.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths','Sqft']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('a', attrs={'class': 'listing-price'}).text.replace('\n', '')
        beds = list.find('div', attrs={'class': 'property-beds'}).text.replace('\n', '')
        baths = list.find('div', attrs={'class': 'property-baths'}).text.replace('\n', '')
        sqft = list.find('div', attrs={'class': 'property-sqft'}).text.replace('\n', '')

        #info["Price"] = pd.to_numeric(info["price"], errors='coerce').fillna(0, downcast='infer')

        info = [price, beds, baths,sqft]
        thewriter.writerow(info)