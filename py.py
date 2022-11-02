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
with open('test.csv', 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Price', 'Beds', 'Baths','Sqft']
    thewriter.writerow(header)

    for list in lists:
        price = list.find('span', attrs={'data-label': 'pc-price'}).text.replace('\n', '')
        beds = list.find('li', attrs={'data-label': 'pc-meta-beds'}).text.replace('\n', '')
        baths = list.find('li', attrs={'data-label': 'pc-meta-baths'}).text.replace('\n', '')
        sqft = list.find('li', attrs={'data-label': 'pc-meta-sqft'})
x = 0
bedrooms = []
bathrooms = []
sqft = []
price = []
p = list(range(40))
for v in p:
    if len(soup.findAll('div', {'class' : 'jsx-11645185 summary-wrap'})[x].contents) == 3:
        try:
            price.append(float(soup.findAll('span', {'data-label' : 'pc-price'})[x].contents[0].replace('$', '').replace(',', '')))
            bedrooms.append(float(soup.findAll('li', {'data-label' : 'pc-meta-beds'})[x].contents[0].contents[0]))
            bathrooms.append(float(soup.findAll('ul', {'data-label' : 'pc-meta-baths'})[x].contents[1].contents[0]))
            sqft.append(float(soup.findAll('ul', {'data-label' : 'pc-meta-sqft'})[x].contents[2].contents[0].replace(',', '')))
            x = x + 1
        except:
            if len(price) > len(bedrooms):
                bedrooms.append(None)
                bathrooms.append(float(soup.findAll('ul', {'data-label' : 'pc-meta-baths'})[x].contents[1].contents[0]))
                sqft.append(float(soup.findAll('ul', {'data-label' : 'pc-meta-sqft'})[x].contents[2].contents[0].replace(',', '')))
                x = x + 1
            elif len(bedrooms) > len(bathrooms):
                bathrooms.append(None)
                sqft.append(float(soup.findAll('ul', {'data-label' : 'pc-meta-sqft'})[x].contents[2].contents[0].replace(',', '')))
                x = x + 1
            else:
                sqft.append(None)
                x = x + 1
            continue
    else:
        x = x + 1


        info = [price, beds, baths,sqft]
        thewriter.writerow(info)