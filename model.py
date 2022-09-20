from warnings import filterwarnings
from pickle import dump
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split

filterwarnings("ignore")

pd.set_option('display.max_columns', None)
train = pd.read_csv('HousingData.csv')


def replacemissing(df):
    Q = pd.DataFrame(df.isna().sum(), columns=['count_mv'])
    R = Q[Q.count_mv > 0]
    cat = []
    con = []
    for i in R.index:
        if df[i].dtypes == 'object':
            replacer = df[i].mode()[0]
            df[i] = df[i].fillna(replacer)
        else:
            replacer = round(df[i].mean(), 2)
            df[i] = df[i].fillna(replacer)


replacemissing(train)

X = train[["LSTAT", "RM", "AGE", "CRIM", "B", "DIS"]]
Y = train[['MEDV']]  # Price
ls = LassoCV(alphas=None, max_iter=200000, normalize=True)
ls = ls.fit(X, Y)

rr = Lasso(alpha=ls.alpha_)
model = rr.fit(X, Y)
xn = [4.32, 6.595, 21.8, 0.02763, 395.63, 5.4011]

# dump(model, open("bostMdl.pkl", "wb"))

print(X.head(50))
print(Y.head(50))
