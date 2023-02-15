import scrapy

class RealtorSpider(scrapy.Spider):
    name = "realtor"
    start_urls = [
        'https://www.realtor.com/realestateandhomes-search/San-Francisco_CA',
    ]

def parse(self, response):
    for listing in response.css('.component_property-card'):
        address = listing.css('.property-address::text').get()
        price = listing.css('.data-price::text').get()
        bedrooms = listing.css('.data-beds::text').get()
        yield {
            'address': address,
            'price': price,
            'bedrooms': bedrooms,
        }

class Property(scrapy.Item):
    address = scrapy.Field()
    price = scrapy.Field()
    bedrooms = scrapy.Field()

import csv

class CsvPipeline:
    def open_spider(self, spider):
        self.file = open('properties.csv', 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=['address', 'price', 'bedrooms'])
        self.writer.writeheader()

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(item)
        return item
