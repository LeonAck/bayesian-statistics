import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

# Load master data
master = pd.read_csv("Data/master.csv")

# Convert date column to datetime
master.date = pd.to_datetime(master.date, format='%Y-%m-%d')


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


# Graph of number of COVID cases per age group
plt.plot(master.date, master.Cases_0_9, label='Agegroup 0-9')
plt.plot(master.date, master.Cases_10_19, label='Agegroup 10-19')
plt.plot(master.date, master.Cases_20_29, label='Agegroup 20-29')
plt.plot(master.date, master.Cases_30_39, label='Agegroup 30-39')
plt.plot(master.date, master.Cases_40_49, label='Agegroup 40-49')
plt.plot(master.date, master.Cases_50_59, label='Agegroup 50-59')
plt.plot(master.date, master.Cases_60_69, label='Agegroup 60-69')
plt.plot(master.date, master.Cases_70_79, label='Agegroup 70-79')
plt.plot(master.date, master.Cases_80_89, label='Agegroup 80-89')
plt.plot(master.date, master.Cases_90_Plus, label='Agegroup 90+')
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





# Test to obtain smoothing spline of ICU inflow
# Not sure if I am doing this right
x = np.linspace(0, 197, 198)
y = master.ICU_Inflow
ss = UnivariateSpline(x, y, k=3) #k=3 means cubic spline
xs = np.linspace(0, 197, 198)
plt.plot(master.date, y, 'ro', ms=5)
plt.plot(master.date, ss(xs), 'g')
plt.show()