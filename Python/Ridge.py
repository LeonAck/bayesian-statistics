"""
Perform a Ridge regression
Op basis van deze link https://machinelearningmastery.com/ridge-regression-with-python/
"""

# Import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load data
master = pd.read_csv("Data/master.csv")

# set date as index of frame
master = master.set_index("date")

# Remove all data up to and including 29-10-2020 as variables contain nan
# This is a temporary solution. We might later decide to exclude these variables
# to use the observations before 29-10-2020 as well
master = master.loc['2021-04-19':'2020-10-30']

# Remove 31-12-2020 since no RNA measurements were at that day
master = master.drop('2020-12-31', axis=0)

# Turn data in to numpy ndarray
data = master.values

# We first standardize the data. Otherwise, different features have different
# penalties
# Create object of StandardScaler class
scaler = StandardScaler()

# Perform standardization
data = scaler.fit_transform(data)

# Create X and y data objects. y is ICU inflow
# Remove ICU column to create X
X = np.delete(data, 0, axis=1)

# Keep ICU column for Y
y = data[:, 0]

# Define model
model = Ridge(alpha=1.0)

# Define model evaluation method. We use 10 fold with three repeats.
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = np.absolute(scores)

print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Nog niet naar estimates gekeken of predictions gemaakt. Nuttig om voor
# verschillende lambda's te doen en te optimizen voor lambda. staat ook in link