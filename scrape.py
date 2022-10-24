from bs4 import BeautifulSoup
import requests
from csv import writer
from urllib.request import urlopen
url= "https://www.century21.com/real-estate/orlando-fl/LCFLORLANDO/"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
lists = soup.find_all('div', class_="property-card-primary-info")

with open('housing.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths', 'Sqft']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('a', class_="listing-price").text.replace('\n', '')
        beds = list.find('div', class_="property-beds").text.replace('\n', '')
        baths = list.find('div', class_="property-baths").text.replace('\n', '')
        sqft = list.find('div', class_="property-sqft").text.replace('\n', '')
        
        info = [price, beds, baths, sqft]
        thewriter.writerow(info)