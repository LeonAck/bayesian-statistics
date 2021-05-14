#### First, we load Python modules
import pandas as pd
import numpy as np
import sklearn as skl

#### Second, we load the data
master = pd.read_csv("Data/master.csv") # load the data
pd.DataFrame.head(master) # view the top n rows of the dataframe
names = list(master) # extract the variable names in a list

#### Third, we use the sklearn package to model the data
# Firstly, we model the elastic net
from sklearn.linear_model import ElasticNetCV
X = master.iloc[:,9:40]
y = master.iloc[:,1]
regr = ElasticNetCV()
regr.fit(X,y)
regr.predict(X)