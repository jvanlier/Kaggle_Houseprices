Kaggle's House Prices: Advanced Regression Techniques
============================================

This repo contains my solution to Kaggle's 
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
competition. 

I managed to get a top 9% score (136th / 1602) on August 11, 2017. 
I scored 0.11579 on the public leaderboard. I'm not done yet though - I have some ideas to improve this 
further that seem very promising.

*Minor caveat*: the version in this repo is actually overfitting to the training data that 
Kaggle provides during the manual transformation selection procedure. 
Internal crossvalidation scores are better but leaderboard scores are worse. 
I'm using a slightly older version to get the score I mentioned, the version currently on the repo 
only does about 0.119 on the public leaderboard. I'll fix it soon. 

I'm sharing this in the hope that it might help others. It's a pretty 
interesting competition: even though it has been around for a couple of years, the top 500 or so 
is dominated by people that have submitted something in the last 2 months. Competition is pretty 
fierce: people are constantly figuring out ways to improve their score.

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
