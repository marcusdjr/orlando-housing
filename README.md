
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

### Visualizing the location of the houses based on latitude and longitude
![longvslat_florida](https://user-images.githubusercontent.com/31329300/213814201-33939b1c-7417-4ee5-96cf-6ed9c0844379.png)
### How Sqft is affecting the sale price of homes
![SqftVIS](https://user-images.githubusercontent.com/31329300/200997466-a7ac5424-2bb4-4296-b806-dbb3f72b7fa5.png)
### How Number of Bedrooms is affecting the sale price of homes
![bedsprice](https://user-images.githubusercontent.com/31329300/200997859-bddab4ac-0427-4077-99a5-de06a9891839.png)

### How the year the house was built affects the homes sale price
![yearbuiltVis](https://user-images.githubusercontent.com/31329300/200997484-2dd4af54-74f0-43e2-8411-1614cb5971c5.png)

## Results

### My Model (Score: 0.8992632194250129)
![Sale Price Accuracy score](https://user-images.githubusercontent.com/31329300/220711508-e2520c1d-05f1-469a-baf6-3bf8bba23054.png)

## Explanation 

In "My Model" here we see an R Squared score of 0.89 
