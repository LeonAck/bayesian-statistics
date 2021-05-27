import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import copy
import seaborn as sns
from scipy.interpolate import UnivariateSpline

"""
eerst log dan first difference van regressors"""
# Load data
data = pd.read_csv("Data/master.csv")

"""
for column in data.columns:
    try:
        data[column] = np.log(data[column])
        print(data[column])
    except (ValueError, AttributeError, TypeError):
        pass
"""

# Convert date column to datetime
data.date = pd.to_datetime(data.date, format='%Y-%m-%d')
data.date = pd.to_datetime(data.date, format='%Y-%m-%d')

# Describe the data
data[['ICU_Inflow', 'RNA', 'Hosp_Inflow', 'Cases', 'Vacc_Est']].describe()

# sort data frame
# data = data.sort_values(by='date', ascending=True)
# data_no_diff = copy.deepcopy(data)
# data_no_diff.sort_vlaues(by='date', ascending=True)
# take first difference of data except for date column
# for column in data.columns[1:]:
 #   data[column] = data[column].diff()

# compare cases to ICU inflow
plt.plot(data.date, data.ICU_Inflow, label='ICU Admissions')
plt.plot(data.date, data.Cases/100, label='positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# compare percentage cases to cases
plt.plot(data.date, np.log(data.Cases), label='Cases')
plt.plot(data.date, np.log(data.Cases_Pct_SMA7d)*4, label='Percentage of positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# compare hospital to ICU inflow
plt.plot(data.date, data.ICU_Inflow_SMA7d*5, label='ICU Admissions')
plt.plot(data.date, data.Hosp_Inflow_SMA7d, label='Hospital admissions')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# compare vacc to ICU inflow
plt.plot(data.date, np.log(data.ICU_Inflow_SMA7d)*4, label='ICU Admissions')
plt.plot(data.date, np.pad(data.Vacc_Est, (0, 1), 'constant'), label='estimated vaccinations')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# compare RNA to ICU inflow
plt.plot(data.date, data.ICU_Inflow_SMA7d, label='ICU Admissions')
plt.plot(data.date, data.RNA_SMA7d/1000000000000, label='RNA')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# Graph of empirical distribution of ICU Inflow
sns.distplot(data.ICU_Inflow)
plt.show()

# Graph of ICU Intake
plt.plot(data.date, data.ICU_Inflow, label='ICU Admissions')
plt.plot(data.date, data.ICU_Inflow_SMA7d, 'r--', label='Moving Average 7 Days')
plt.plot(data.date, data.ICU_Inflow_SMA14d, 'k-.', label='Moving Average 14 Days')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of Hospital Intake
plt.plot(data.date, data.Hosp_Inflow, label='Hospital Admissions')
plt.plot(data.date, data.Hosp_Inflow_SMA7d, 'r--', label='Moving Average 7 Days')
plt.plot(data.date, data.Hosp_Inflow_SMA14d, 'k-.', label='Moving Average 14 Days')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of number of COVID cases
plt.plot(data.date, data.Cases, label='Number of cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# Graph of percentage of positive tests
plt.plot(data.date, data.Cases_Pct, label='Percentage of positive cases')
plt.plot(data.date, data.Cases_Pct_SMA3d, 'r--', label='Moving Average 3 Days')
plt.plot(data.date, data.Cases_Pct_SMA7d, 'k-.', label='Moving Average 7 Days')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# Graph of prevalence
plt.plot(data.date, data.Prev, label='Number of infectious people')
plt.xlabel('Date')
plt.ylabel('Infectious People')
plt.legend()
plt.show()

# Graph of R number
plt.plot(data.date, data.R, label='Reproduction Number')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()
plt.show()

# Graph of RNA
plt.plot(data.date, data.RNA, label='RNA in waste water')
plt.plot(data.date, data.RNA_SMA3d, 'r--', label='Moving Average 3 Days')
plt.plot(data.date, data.RNA_SMA7d, 'k-.', label='Moving Average 7 Days')
plt.xlabel('Date')
plt.ylabel('RNA')
plt.legend()
plt.show()

# Compare percentage cases to ICU inflow
plt.plot(data.date, data.ICU_Inflow_SMA7d, label='ICU Admissions')
plt.plot(data.date, data.Cases_Pct_SMA7d*2.5, label='Percentage of positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

"""
# Test to obtain smoothing spline of ICU inflow
# Not sure if I am doing this right
x = np.linspace(0, 197, 198)
y = data.ICU_Inflow
ss = UnivariateSpline(x, y, k=3) # k=3 means cubic spline
xs = np.linspace(0, 197, 198)
plt.plot(data.date, y, 'ro', ms=5)
plt.plot(data.date, ss(xs), 'g')
plt.show()
"""
