
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

## Results

# My Model (Score: 0.9758446486618788)

#
# VS

# Model 1 (Score: 0.9526462263899598)
![VIS2](https://user-images.githubusercontent.com/31329300/200976347-6cc8ffcc-a08b-40ac-ae37-108e05b7f8b6.png)

## Explanation 

In "My Model" here we see an R Squared score of 0.97 while in "Model 1" we see an R Squared score of 0.95 meaning that "My Model" is more accurate when it comes to predicting the sale price of homes than "Model" 1 is. "Model 1" predicts the sale price of homes based off the price the home was listed for and we see that there is a %95 chance of it getting the prediction right but on the other hand "My Model" does %2 better than "Model 1" making it the better model. 
