
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

# Most common bedroom number
![BedroomsVIS](https://user-images.githubusercontent.com/31329300/200997421-ba548faa-d64a-48c1-8b70-0aabc24a6351.png)
# Visualizing the location of the houses based on latitude and longitude
![LatLongVis](https://user-images.githubusercontent.com/31329300/200997459-29b7d859-f671-478e-b5f9-2eaafa72f8f4.png)
# How Sqft is affecting the sale price of homes
![SqftVIS](https://user-images.githubusercontent.com/31329300/200997466-a7ac5424-2bb4-4296-b806-dbb3f72b7fa5.png)
# How Location is affecting the sale price of homes
![Location Vis](https://user-images.githubusercontent.com/31329300/200997473-243814df-e8ef-4df2-81b6-586caee2fcda.png)
# How Number of Bedrooms is affecting the sale price of homes
![bedsprice](https://user-images.githubusercontent.com/31329300/200997859-bddab4ac-0427-4077-99a5-de06a9891839.png)
# How the amount of days home was on zillow affected the sale price
![DaysVIS](https://user-images.githubusercontent.com/31329300/200997481-2f6d9c89-fd6c-4ee1-a1bb-3ebec98a6e4c.png)
# How the year the house was built affects the homes sale price
![yearbuiltVis](https://user-images.githubusercontent.com/31329300/200997484-2dd4af54-74f0-43e2-8411-1614cb5971c5.png)

## Results

# My Model (Score: 0.9758446486618788)
![VIS1](https://user-images.githubusercontent.com/31329300/201108108-e80ba8ad-f596-4a76-9abc-1c2a8c99e0d3.png)

#
# VS
#

# Model 1 (Score: 0.9526462263899598)
![VIS2](https://user-images.githubusercontent.com/31329300/200976347-6cc8ffcc-a08b-40ac-ae37-108e05b7f8b6.png)

## Explanation 

In "My Model" here we see an R Squared score of 0.97 while in "Model 1" we see an R Squared score of 0.95 meaning that "My Model" is more accurate when it comes to predicting the sale price of homes than "Model" 1 is. "Model 1" predicts the sale price of homes based off the price the home was listed for and we see that there is a %95 chance of it getting the prediction right but on the other hand "My Model" does %2 better than "Model 1" making it the better model. 
