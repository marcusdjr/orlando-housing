from bs4 import BeautifulSoup
import requests
from csv import writer

url= "https://www.century21.com/real-estate/orlando-fl/LCFLORLANDO/"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
lists = soup.find_all('div', class_="property-card-primary-info")

with open('housing.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Title', 'Location', 'Price', 'Area']
    thewriter.writerow(header)

    for list in lists:

        price = list.find('a', class_="listing-price").text.replace('\n', '')
        beds = list.find('div', class_="property-beds").text.replace('\n', '')
        
        info = [price, beds]
        thewriter.writerow(info)