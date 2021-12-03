# House Prices - Advanced Regression Techniques 

![](img/header.png)

<br>

# 1. Overview

_"Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home."_

See on [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)


<br>


# 2. The problem

 **1. Context:**

   A startup **SweetHome**, begins his operation of real state in Ames, Iowa. The main business is to buy and sell houses, like the [Zillow](https://www.zillow.com/).

 **2.The Bussiness Problem:**
   
   A startup needs to evaluate the houses correctly, so they can buy and sell assuring a profit margin and the manutention of the operation. 

   The **SweetHome** has access to a dataset with houses prices in Ames, then the CTO decided that they can use Machine Learning models to forecast the house prices. You and your team were designated to build the best model for this challenge.

 **3. Stakeholders:**

   The CTO wants a model that can do the forecasts at the least cost possible.
    
   The business team will use the model forecast to help in the evaluation of the home.

   The engineering team needs to know how much one feature increase the house's prices, to estimate if reform is viable and profitable.

   The development team needs a deployable model, lightweight, and scalable model.
   

<br>


# 3. Data

The dataset has 79 explanatory variables, of homes aspects to predict the final price.

* SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
* MSSubClass: The building class.
* MSZoning: The general zoning classification.
* LotFrontage: Linear feet of street connected to property.
* LotArea: Lot size in square feet.
* Street: Type of road access.
* Alley: Type of alley access.
* LotShape: General shape of property.
* LandContour: Flatness of the property.
* Utilities: Type of utilities available.
* LotConfig: Lot configuration.
* LandSlope: Slope of property.
* Neighborhood: Physical locations within Ames city limits.
* Condition1: Proximity to main road or railroad.
* Condition2: Proximity to main road or railroad (if a second is present).
* BldgType: Type of dwelling.
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality.
* OverallCond: Overall condition rating.
* YearBuilt: Original construction date.
* YearRemodAdd: Remodel date.
* RoofStyle: Type of roof.
* RoofMatl: Roof material.
* Exterior1st: Exterior covering on house.
* Exterior2nd: Exterior covering on house (if more than one material).
* MasVnrType: Masonry veneer type.
* MasVnrArea: Masonry veneer area in square feet.
* ExterQual: Exterior material quality.
* ExterCond: Present condition of the material on the exterior.
* Foundation: Type of foundation.
* BsmtQual: Height of the basement.
* BsmtCond: General condition of the basement.
* BsmtExposure: Walkout or garden level basement walls.
* BsmtFinType1: Quality of basement finished area.
* BsmtFinSF1: Type 1 finished square feet.
* BsmtFinType2: Quality of second finished area (if present).
* BsmtFinSF2: Type 2 finished square feet.
* BsmtUnfSF: Unfinished square feet of basement area.
* TotalBsmtSF: Total square feet of basement area.
* Heating: Type of heating.
* HeatingQC: Heating quality and condition.
* CentralAir: Central air conditioning.
* Electrical: Electrical system.
* 1stFlrSF: First Floor square feet.
* 2ndFlrSF: Second floor square feet.
* LowQualFinSF: Low quality finished square feet (all floors).
* GrLivArea: Above grade (ground) living area square feet.
* BsmtFullBath: Basement full bathrooms.
* BsmtHalfBath: Basement half bathrooms.
* FullBath: Full bathrooms above grade.
* HalfBath: Half baths above grade.
* Bedroom: Number of bedrooms above basement level.
* Kitchen: Number of kitchens.
* KitchenQual: Kitchen quality.
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms).
* Functional: Home functionality rating.
* Fireplaces: Number of fireplaces.
* FireplaceQu: Fireplace quality.
* GarageType: Garage location.
* GarageYrBlt: Year garage was built.
* GarageFinish: Interior finish of the garage.
* GarageCars: Size of garage in car capacity.
* GarageArea: Size of garage in square feet.
* GarageQual: Garage quality.
* GarageCond: Garage condition.
* PavedDrive: Paved driveway.
* WoodDeckSF: Wood deck area in square feet.
* OpenPorchSF: Open porch area in square feet.
* EnclosedPorch: Enclosed porch area in square feet.
* 3SsnPorch: Three season porch area in square feet.
* ScreenPorch: Screen porch area in square feet.
* PoolArea: Pool area in square feet.
* PoolQC: Pool quality.
* Fence: Fence quality.
* MiscFeature: Miscellaneous feature not covered in other categories.
* MiscVal: $Value of miscellaneous feature.
* MoSold: Month Sold.
* YrSold: Year Sold.
* SaleType: Type of sale.
* SaleCondition: Condition of sale.

<br>

# 4. Approach

The steps to solve this problem were:

**1. Modeling the problem:** 

Because the houses prices is a real number, the problem can be modeled as an regression.

**2. Understanding the variables:** 

By the exploratory data analysis, we will take a look at the predictors (houses features), and the target variable (sale price) to see what are their relationship, how much missing that are in dataset, and the types of the variables.

**3. Data Preparation:** 

Is this step the data was cleaned, encoded, and filled. This step aims to make data usable by machine learning models.

**4. Model Baseline:** 

A baseline model was created using just the Data Preparation step.

**5. Feature Engineering:** 

New features was created, and those that improve the baseline model were kept in the database.

**6. Final Model:** 

Machine Learning model training.

**7. Tunning Hyperparameter:** 

Select the best model parameters to achieve the best score.

<br>

