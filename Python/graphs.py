import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import csaps
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

# get 7 day SMA for cases

# Describe the data
print(data[['ICU_Inflow', 'RNA', 'Hosp_Inflow', 'Cases', 'Vacc_Est']].describe())


# graph 2 data exploration
plt.plot(data.date, data.ICU_Inflow_SMA7d, label='ICU Admissions', linewidth=3)
plt.plot(data.date, data.Cases_SMA7d/125, '--', label='Positive Cases / 125 ')
plt.plot(data.date, data.Cases_Pct_SMA7d*2.5, label='Percentage of positive cases * 2.5')
plt.plot(data.date, data.Tested_SMA7d/1000, '--', label='Tests/ 1,000')
plt.xlabel('Date')
plt.ylabel('Value')
axes = plt.gca()
axes.set_ylim([0, 100])
plt.legend()
plt.savefig("Images/Data_exploration_2.png")
plt.show()

# graph 1 data exploration
plt.plot(data.date, data.ICU_Inflow_SMA7d, label='ICU Admissions', linewidth=3)
plt.plot(data.date, data.Hosp_Inflow_SMA7d/5, '--', label='Hospital Admissions / 5')
plt.plot(data.date, data.RNA_SMA7d/1000000000000, '--', label='RNA per Measuring Station / 10^12')
plt.xlabel('Date')
plt.ylabel('Value')
axes = plt.gca()
axes.set_ylim([0, 100])
plt.legend()
plt.savefig("Images/Data_exploration_1.png")
plt.show()

# compare cases to ICU inflow
plt.plot(data.date, data.ICU_Inflow, label='ICU Admissions')
plt.plot(data.date, data.Cases/125, label='positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# graph over shorter time period
plt.plot(data.date.loc[129:], data.ICU_Inflow.loc[129:], label='ICU Admissions')
plt.plot(data.date.loc[129:], data.Cases.loc[129:]/100, '-.', label='Cases')
plt.xlabel('Date')
plt.ylabel('number')
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

# compare # tests to ICU inflow
plt.plot(data.date, data.ICU_Inflow_SMA7d, label='ICU Admissions')
plt.plot(data.date, data.Tested_SMA7d/1000, label='Tests')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# graph over shorter time period
plt.plot(data.date.loc[129:], data.ICU_Inflow.loc[129:]*5, label='ICU Admissions')
plt.plot(data.date.loc[129:], data.Hosp_Inflow.loc[129:], '-.', label='Hospital admissions')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

"""
# compare vacc to ICU inflow
plt.plot(data.date, np.log(data.ICU_Inflow_SMA7d)*4, label='ICU Admissions')
plt.plot(data.date, np.pad(data.Vacc_Est, (0, 1), 'constant'), label='estimated vaccinations')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()
"""
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
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of Hospital Intake
plt.plot(data.date, data.Hosp_Inflow, label='Hospital Admissions')
plt.plot(data.date, data.Hosp_Inflow_SMA7d, 'r--', label='Moving Average 7 Days')
# plt.plot(data.date, data.Hosp_Inflow_SMA14d, 'k-.', label='Moving Average 14 Days')
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


# Test to obtain smoothing spline of ICU inflow
# Not sure if I am doing this right
x = np.linspace(0, 191, 192)
y = data.ICU_Inflow
ss = UnivariateSpline(x, y, k=3) # k=3 means cubic spline
xs = np.linspace(0, 191, 192)
# ss.set_smoothing_factor(0.5)
plt.plot(data.date, y, 'ro', ms=5)
plt.plot(data.date, ss(xs), 'g')
plt.show()
