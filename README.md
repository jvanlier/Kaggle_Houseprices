Kaggle's House Prices: Advanced Regression Techniques
============================================

This repo contains my solution to Kaggle's 
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
competition. 

I managed to get a top 5% score on August 16, 2017 with a score of .11459.

The writeup is on [my blog](http://jvlanalytics.nl/blog/2017/08/26/kaggle-house-price-prediction.html). 

The code was written (and has only been tested) on a Mac using Anaconda Python 3.6. See `requirements.txt` for the modules used and their versions.

# Files in repo
- `Explorative_Data_Analysis.ipynb`: Jupyter notebook which shows how I analyzed the data, 
including observations and conclusions.
- `Model.ipynb`: Jupyter notebook with machine learning code.
- `crossval.py`: Cross-validation helper functions.
- `preprocess.py`: Data pre-processing functions, K-Nearest Neighbour imputation.
- `utils.py`: Various functions for scoring metrics, numeric transformations, plots etc.

# Running it yourself
Get the data from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and place it in a directory called `data`. Install the pre-requisite packages and fire up 
`jupyter notebook` using a Python 3.6 kernel.
