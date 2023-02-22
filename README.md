
# 


[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

![Orlando-Florida-aerial-cityscape-towards-Eola-Lake](https://user-images.githubusercontent.com/31329300/198492770-4db960a5-9384-4cc4-802d-5d468207a0d4.png)

>In recent times our economy has been of great concern, inflation has caused rises in prices like we have never seen before especially when it comes to the cost of homes. I have been very interested in purchasing a home with my family in the Orlando Florida area but inflation has made the search difficult. My goal with this project is to estimate the sales price for each house within a 20 miles radius of Orlando Florida so that I know exactly what a "Good Price" looks like when purchasing my home.

## Process

- Scrape the data from each home on the market within a 20 mile radius of Orlando Florida. 
- Use ploomber to build a data pipeline. 
- Use GitHub Actions to run the scarper every day and pull in newly added homes.
- Estimate the sale price of homes within a 20 mile radius of Orlando Florida.

## Analysis

### How Sqft is affecting the sale price of homes
![SqftVIS](https://user-images.githubusercontent.com/31329300/220726343-212724a7-1add-4d79-9995-c9ac8f5edb61.png)
### How Number of Bedrooms is affecting the sale price of homes
![VIS1](https://user-images.githubusercontent.com/31329300/220730981-fa6bf95b-a86d-41a2-90af-1a64237e7f07.png)
### How Number of Bathrooms is affecting the sale price of homes
![VIS2](https://user-images.githubusercontent.com/31329300/220731572-a68ec7bc-ad04-48de-a764-0569982d9ede.png)


## Results

### My Model (Score: 0.8992632194250129)
![Sale Price Accuracy score](https://user-images.githubusercontent.com/31329300/220711508-e2520c1d-05f1-469a-baf6-3bf8bba23054.png)

## Explanation 

In My Model here we see an R Squared score of 0.8992632194250129
