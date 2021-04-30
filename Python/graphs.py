import matplotlib.pyplot as plt
import datetime as dt


plt.scatter(data_mzelst.date, data_mzelst.IC_Intake, s=2, label='IC Admissions')
plt.scatter(data_mzelst.date, data_mzelst.Hospital_Intake, s=2, label='Hospital Admissions')
plt.scatter(data_mzelst.date, data_mzelst.positivetests/25, s=2, label='Positive Tests/25')
#plt.axvline(dt.datetime(2020, 3, 12), color='k', lw='1')
#plt.axvline(dt.datetime(2020, 3, 23), color='k', lw='1')
plt.xlabel('Date')
plt.legend()
plt.rcParams["figure.figsize"] = (8,4)
plt.rcParams["legend.loc"] = 'best'
plt.savefig('Images/graph.png')
plt.show()

plt.scatter(data_)