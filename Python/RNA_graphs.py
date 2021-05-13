import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
I believe this file is no longer up to date
    - Roel
"""

# Load data
grouped_rna_df = pd.read_csv("Data/grouped_rna.csv", index_col=0)
data_mzelst = pd.read_csv("Data/mzelst.csv")

# Prepare data
data_mzelst.date = pd.to_datetime(data_mzelst.date, format='%Y-%m-%d')
data_mzelst = data_mzelst.drop(427, axis=0)

# plot scatter no moving average
plt.scatter(grouped_rna_df.Date_measurement, grouped_rna_df.RNA_flow_per_100000,
            s=2, label='RNA per 100000 inhabitants')
plt.legend()
plt.savefig('Images/RNA_per_100000_grouped.png')
plt.show()
# plot moving average per three days
plt.scatter(grouped_rna_df.Date_measurement, grouped_rna_df.pandas_SMA_3,
            s=2, label='Three day moving averages RNA per 1000000')
plt.legend()
plt.savefig('Images/RNA_per_100000_SMA_3.png')
plt.show()
# scatter moving average per seven days
plt.scatter(grouped_rna_df.Date_measurement, grouped_rna_df.pandas_SMA_7,
            s=2, label='Seven day moving averages RNA per 1000000')
plt.legend()
plt.savefig('Images/RNA_per_100000_SMA_7.png')
plt.show()

# plot line in three day moving average
# plt.plot(grouped_rna_df.Date_measurement, grouped_rna_df.pandas_SMA_3,
# label='IC Admissions')
# plt.show()
# plot line in seven day moving average
plt.plot(grouped_rna_df.Date_measurement, grouped_rna_df.pandas_SMA_7 / (4 * 10 ** 13),
         label='Seven day moving average')
plt.scatter(data_mzelst.date, data_mzelst.IC_Intake, s=2, label='IC Admissions',
            color='green')
plt.legend()
plt.show()
plt.savefig('Images/comparison_IC_RNA.png')
