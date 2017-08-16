Kaggle's House Prices: Advanced Regression Techniques
============================================

This repo contains my solution to Kaggle's 
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
competition. 

I managed to get a top 5% score (76th / 1584) on August 16, 2017 with a score of .11459.
I think I'll leave it at this - I've invested all the time that I wanted to (and more).

I'm sharing this in the hope that it might help others. It's a pretty 
interesting competition: even though it has been around for a couple of years, the top 500 or so 
is dominated by people that have submitted something in the last 2 months. Competition is quite 
fierce: people are constantly figuring out ways to improve their score.

A short blogpost with lessons learned is coming soon!

The code was written (and has only been tested) on a Mac using Anaconda Python 3.6. See `requirements.txt` for the modules used and their versions.

# Files in repo
- `Explorative_Data_Analysis.ipynb`: Jupyter notebook which shows how I analyzed the data, 
including observations and conclusions.
- `ML.ipynb`: Jupyter notebook with machine learning code.
- `crossval.py`: Cross-validation helper functions.
- `preprocess.py`: Data pre-processing functions, K-Nearest Neighbour imputation.
- `utils.py`: Various functions for scoring metrics, numeric transformations, plots etc.

# Running it yourself
Get the data from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and place it in a directory called `data`. Install the pre-requisite packages and fire up 
`jupyter notebook` using a Python 3.6 kernel.
