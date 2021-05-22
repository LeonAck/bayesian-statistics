import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import copy
from scipy.interpolate import UnivariateSpline

"""
eerst log dan first difference van regressors"""
# Load master data
master = pd.read_csv("Data/master.csv")
# replace date where vaccin columns are all zero with nan
master.loc[[21], ['Vacc_Est', 'Vacc_Est_Carehomes', 'Vacc_Adm_GGD',
            'Vacc_Adm_Hosp', 'Vacc_Adm_Doctors']] = np.NAN

# interpolate nans in RNA data and in vaccin columns
master = master.interpolate()
"""
for column in master.columns:
    try:
        master[column] = np.log(master[column])
        print(master[column])
    except (ValueError, AttributeError, TypeError):
        pass
"""
# Convert date column to datetime
master.date = pd.to_datetime(master.date, format='%Y-%m-%d')

# Describe the data


# sort data frame
# master = master.sort_values(by='date', ascending=True)
# master_no_diff = copy.deepcopy(master)
# master_no_diff.sort_vlaues(by='date', ascending=True)
# take first difference of data except for date column
#for column in master.columns[1:]:
 #   master[column] = master[column].diff()

# compare cases to ICU inflow
plt.plot(master.date, master.ICU_Inflow, label='ICU Admissions')
plt.plot(master.date, master.Cases/100, label='positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# compare percentage cases to cases
plt.plot(master.date, np.log(master.Cases), label='Cases')
plt.plot(master.date, np.log(master.Cases_Pct_SMA7d)*4, label='Percentage of positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# compare hospital to ICU inflow
plt.plot(master.date, master.ICU_Inflow_SMA7d*4, label='ICU Admissions')
plt.plot(master.date, master.Hosp_Inflow_SMA7d, label='Hospital admissions')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# compare vacc to ICU inflow
plt.plot(master.date, np.log(master.ICU_Inflow_SMA7d)*4, label='ICU Admissions')
plt.plot(master.date, np.log(master.Vacc_Est), label='estimated vaccinations')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# compare RNA to ICU inflow
plt.plot(master.date, master.ICU_Inflow_SMA7d, label='ICU Admissions')
plt.plot(master.date, master.RNA_SMA7d/1000000000000, label='RNA')
plt.xlabel('Date')
plt.ylabel('number')
plt.legend()
plt.show()

# Graph of ICU Intake
plt.scatter(master.date, master.ICU_Inflow, s=4)
plt.plot(master.date, master.ICU_Inflow, label='ICU Admissions')
plt.plot(master.date, master.ICU_Inflow_SMA7d, 'r--', label='Moving Average 7 Days')
plt.plot(master.date, master.ICU_Inflow_SMA14d, 'k-.', label='Moving Average 14 Days')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of Hospital Intake
plt.plot(master.date, master.Hosp_Inflow, label='Hospital Admissions')
plt.plot(master.date, master.Hosp_Inflow_SMA7d, 'r--', label='Moving Average 7 Days')
plt.plot(master.date, master.Hosp_Inflow_SMA14d, 'k-.', label='Moving Average 14 Days')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of number of COVID cases
plt.plot(master.date, master.Cases, label='Number of cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# Graph of percentage of positive tests
plt.plot(master.date, master.Cases_Pct, label='Percentage of positive cases')
plt.plot(master.date, master.Cases_Pct_SMA3d, 'r--', label='Moving Average 3 Days')
plt.plot(master.date, master.Cases_Pct_SMA7d, 'k-.', label='Moving Average 7 Days')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()

# Graph of prevalence
plt.plot(master.date, master.Prev, label='Number of infectious people')
plt.plot(master.date, master.Prev_LB, 'k--', label='Lowerbound')
plt.plot(master.date, master.Prev_UB, 'k--', label='Upperbound')
plt.xlabel('Date')
plt.ylabel('Infectious People')
plt.legend()
plt.show()

# Graph of R number
plt.plot(master.date, master.R, label='Reproduction Number')
plt.plot(master.date, master.R_LB, 'k--', label='Lowerbound')
plt.plot(master.date, master.R_UB, 'k--', label='Upperbound')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()
plt.show()

# Graph of RNA
plt.plot(master.date, master.RNA, label='RNA in waste water')
plt.plot(master.date, master.RNA_SMA3d, 'r--', label='Moving Average 3 Days')
plt.plot(master.date, master.RNA_SMA7d, 'k-.', label='Moving Average 7 Days')
plt.xlabel('Date')
plt.ylabel('RNA')
plt.legend()
plt.show()

# Compare percentage cases to ICU inflow
plt.plot(master.date, master.ICU_Inflow_SMA7d, label='ICU Admissions')
plt.plot(master.date, master.Cases_Pct_SMA7d*2.5, label='Percentage of positive cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()


# Test to obtain smoothing spline of ICU inflow
# Not sure if I am doing this right
x = np.linspace(0, 197, 198)
y = master.ICU_Inflow
ss = UnivariateSpline(x, y, k=3) # k=3 means cubic spline
xs = np.linspace(0, 197, 198)
plt.plot(master.date, y, 'ro', ms=5)
plt.plot(master.date, ss(xs), 'g')
plt.show()
