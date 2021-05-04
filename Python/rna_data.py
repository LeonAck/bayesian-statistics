import pandas as pd
from collections import Counter
import numpy as np

# load data
data_rivm_riool = pd.read_csv("Data/rivm_rioolwaterdata.csv", sep=';')

# Transform date column from string to datetime
data_rivm_riool.Date_measurement = pd.to_datetime(data_rivm_riool.Date_measurement, format='%Y-%m-%d')

# Sort dataframes to get newest date first
data_rivm_riool = data_rivm_riool.sort_values(by='Date_measurement', ascending=False)


################ DATA TRANSFORMATION / CREATION #######################3
"""In this section, we create a few new variables and save them in the dataframe
 'grouped_rna_df'
- RNA_per_ml summed per day
- RNA_per_100000 summed per day
- Moving average for three and seven days for RNA_per_100000 summed_per_day
- Number of measurements per day
- RNA_per_100000 summed per day / number of measurements per day
- MA for 3 and 7 days for RNA_per_1000000 per measurement 
"""

# Group RNA measurements by date
grouped_rna_per_100000 = data_rivm_riool.groupby("Date_measurement")[
    'RNA_flow_per_100000'].sum()
grouped_rna_per_ml = data_rivm_riool.groupby("Date_measurement")[
    'RNA_per_ml'].sum()

# count number of measurements by day
measurement_count = data_rivm_riool.groupby(
    "Date_measurement")[
    'RNA_flow_per_100000'].count()

# rename series
measurement_count = measurement_count.rename("Measurement_count")

# get array of unique dates
unique_dates = data_rivm_riool['Date_measurement'].unique()

# turn array in Series
unique_dates = pd.Series(unique_dates)

# combine dates Series with grouped_rna series
grouped_rna_df = pd.concat([
 grouped_rna_per_100000, grouped_rna_per_ml, measurement_count],
    axis=1)

# make dates a column
grouped_rna_df.reset_index(level=0, inplace=True)

# get SMA for three days
grouped_rna_df['pandas_SMA_3'] = grouped_rna_df.iloc[
                                 :, 1].rolling(window=3).mean()

# get SMA for seven days
grouped_rna_df['pandas_SMA_7'] = grouped_rna_df.iloc[
                                 :, 1].rolling(window=7).mean()

# create new column with rna_per_100000 divided by number of measurements on that day
grouped_rna_df['RNA_per_100000_per_measurement'] = grouped_rna_df[
    'RNA_flow_per_100000'] / grouped_rna_df['Measurement_count']

# save grouped_rna_df to disc
grouped_rna_df.to_csv(r'Data\grouped_rna.csv')

################# DATA INSPECTION #####################################3
# get unique rioolwaterinstallaties
unique_locations = data_rivm_riool['RWZI_AWZI_name'].unique()

# count number of unqique locations
print(len(unique_locations)) # 319 unique locations

# count the nummber of occurrences per location
# no location occurs more than 80 times over
location_counter = Counter(data_rivm_riool['RWZI_AWZI_name'])

# convert to dictionary and then to Df
location_counter = dict(location_counter)
location_counter = pd.DataFrame.from_dict(location_counter, orient='index')
mean_location_counter = np.mean(location_counter)


