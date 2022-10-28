from bs4 import BeautifulSoup
import requests
from requests_html import HTML, HTMLSession
from csv import writer
from urllib.request import urlopen
url= "https://www.realtor.com/realestateandhomes-search/Orlando_FL"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
lists = soup.find_all('div', class_="jsx-11645185 summary-wrap")

with open('listing.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('span', attrs={'data-label': 'pc-price'}).text.replace('\n', '')
        beds = list.find('li', attrs={'data-label': 'pc-meta-beds'}).text.replace('\n', '')
        baths = list.find('li', attrs={'data-label': 'pc-meta-baths'}).text.replace('\n', '')
        
        info = [price, beds, baths]
        thewriter.writerow(info)