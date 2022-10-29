from bs4 import BeautifulSoup
import requests
from requests_html import HTML, HTMLSession
from csv import writer
from urllib.request import urlopen

url= "https://www.century21.com/real-estate/orlando-fl/LCFLORLANDO/"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
lists = soup.find_all('div', class_="property-card-primary-info")

with open('housing.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('div', attrs={'class': 'listing-price'})
        beds = list.find('div', attrs={'class': 'property-beds'}).text.replace('\n', '')
        baths = list.find('div', attrs={'class': 'property-baths'}).text.replace('\n', '')
        
        info = [price, beds, baths]
        thewriter.writerow(info)