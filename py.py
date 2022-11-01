from bs4 import BeautifulSoup
import requests
from requests_html import HTML, HTMLSession
from csv import writer
from urllib.request import urlopen
import pandas as pd

url= "https://www.realtor.com/realestateandhomes-search/Orlando_FL" 

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
lists = soup.find_all('div', class_="jsx-11645185 summary-wrap")

with open('house.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths','Sqft']
    thewriter.writerow(header)

for list in lists:
        price = list.find('div', {'class' : 'list-card-price'}).text.replace('\n', '')
        beds = list.find('ul', {'class' : 'list-card-details'}).text.replace('\n', '')
        baths = list.find('ul', {'class' : 'list-card-details'}).text.replace('\n', '')
        sqft = list.find('ul', {'class' : 'list-card-details'}).text.replace('\n', '')

        #info["Price"] = pd.to_numeric(info["price"], errors='coerce').fillna(0, downcast='infer')

        info = [price, beds, baths,sqft]
        thewriter.writerow(info)


