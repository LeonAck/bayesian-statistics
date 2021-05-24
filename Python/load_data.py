import pandas as pd
import numpy as np

################ LOAD DATA AND COMBINE DATA INTO MASTER DATAFRAME #######################
"""In this section, we load all data sources and create a master dataframe 
containing all variables, saving it in the dataframe 
'master_data'
"""

# Load data
data_ggd_tests = pd.read_csv("Data/ggd_tests_per_agegroup.csv")
data_lcps_admissions = pd.read_csv("Data/lcps_admissions.csv")
data_rivm_prevalence = pd.read_csv("Data/rivm_prevalence.csv")
data_rivm_reproduction = pd.read_csv("Data/rivm_reproduction_number.csv")
data_rivm_tests = pd.read_csv("Data/rivm_tests.csv")
# pass own column names to loading vaccine data to force column names
data_vaccines = pd.read_csv(
    "Data/vaccines.csv",
    header=0, names=["date",
                     "vaccines_administered_estimated_carehomes",
                     "vaccines_administered_ggd",
                     "vaccines_administered_hospital",
                     "vaccines_administered_estimated",
                     "vaccines_administered",
                     "vaccines_administered_doctors"])
data_rna = pd.read_csv("Data/grouped_rna.csv", index_col=0)

# data_rivm_gedrag = pd.read_csv("Data/rivm_gedrag.csv", sep=';')
#   Preparing the gedrag dataset is on hold due to time efficiency
#   There is only data every 3 weeks until april 20th available,
#   so not the best source of data


# change conflicting date variables to "date"
data_rivm_prevalence = data_rivm_prevalence.rename(columns={"Date": "date"})
data_rivm_reproduction = data_rivm_reproduction.rename(columns={"Date": "date"})
data_rna = data_rna.rename(columns={"Date_measurement": "date"})

# Transform date column from string to datetime
data_ggd_tests.date = pd.to_datetime(data_ggd_tests.date, format='%Y-%m-%d')
data_lcps_admissions.date = pd.to_datetime(data_lcps_admissions.date, format='%Y-%m-%d')
data_rivm_prevalence.date = pd.to_datetime(data_rivm_prevalence.date, format='%Y-%m-%d')
data_rivm_reproduction.date = pd.to_datetime(data_rivm_reproduction.date, format='%Y-%m-%d')
data_rivm_tests.date = pd.to_datetime(data_rivm_tests.date, format='%Y-%m-%d')
data_vaccines.date = pd.to_datetime(data_vaccines.date, format='%Y-%m-%d')
data_rna.date = pd.to_datetime(data_rna.date, format='%Y-%m-%d')

# GGD dataset: aggregate data per GGD on a national level per age group
# Also transform from long into wide format (i.e. with only date as a row index,
# and cases_0-9 cases_10-19 etc as column variables)
# ?? There is a "<50" and "unknown" group. What to do with those?
# 1. Group the data by date and age
agg_ggd_tests = data_ggd_tests.groupby(by=["date", "age_group"]).sum()

# 2. Create a national level positive test variable for each age group as DataFrame
agg_ggd_tests_wide = agg_ggd_tests.unstack(level=-1)

# Sort dataframes to get newest date first
agg_ggd_tests_wide = agg_ggd_tests_wide.sort_values(by='date', ascending=False)
data_lcps_admissions = data_lcps_admissions.sort_values(by='date', ascending=False)
data_rivm_prevalence = data_rivm_prevalence.sort_values(by='date', ascending=False)
data_rivm_reproduction = data_rivm_reproduction.sort_values(by='date', ascending=False)
data_rivm_tests = data_rivm_tests.sort_values(by='date', ascending=False)
data_vaccines = data_vaccines.sort_values(by='date', ascending=False)
data_rna = data_rna.sort_values(by='date', ascending=False)

# Drop duplicates vaccine data. For some reason duplicates are created when sorting
data_vaccines = data_vaccines.drop_duplicates(subset='date')

# Variables below have the same value over all observations
data_rivm_prevalence = data_rivm_prevalence.drop(labels=["population", "version"], axis=1)
data_rivm_reproduction = data_rivm_reproduction.drop(labels="version", axis=1)


# Only keep data from 2020-10-17 (start LCPS IC admission dataset)
# 1. Make the date column the index of the dataset.
# This has already been done for agg_ggd_tests_wide
data_lcps_admissions = data_lcps_admissions.set_index("date")
data_rivm_prevalence = data_rivm_prevalence.set_index("date")
data_rivm_reproduction = data_rivm_reproduction.set_index("date")
data_rivm_tests = data_rivm_tests.set_index("date")
data_vaccines = data_vaccines.set_index("date")
data_rna = data_rna.set_index("date")

# 2. Slice all dates to 2020-10-17
data_lcps_admissions = data_lcps_admissions.loc[:'2020-10-17']
data_rivm_prevalence = data_rivm_prevalence.loc[:'2020-10-17']
data_rivm_reproduction = data_rivm_reproduction.loc[:'2020-10-17']
data_rivm_tests = data_rivm_tests.loc[:'2020-10-17']
data_vaccines = data_vaccines.loc[:'2020-10-17']
agg_ggd_tests_wide = agg_ggd_tests_wide.loc[:'2020-10-17']
data_rna = data_rna.loc[:'2020-10-17']


### MASTER DATAFRAME

# Create master dataframe containing all datasets
master = pd.concat([agg_ggd_tests_wide, data_lcps_admissions,
                    data_rivm_prevalence, data_rivm_reproduction,
                    data_rivm_tests, data_vaccines, data_rna], axis=1)

# Sort master dataframe to get oldest data first
master = master.sort_values(by='date', ascending=True)

# Delete superfluous variables
master = master.drop(labels=[('cases', '<50'), ('cases', 'Unknown'),
                             'population', 'IC_Bedden_Non_COVID',
                             'RNA_per_ml', 'RNA_flow_per_100000',
                             'Measurement_count',
                             'groei_besmettelijken', 'besmet_7daverage'
                             ], axis=1)

# Rename variables for clarity and consistency
master = master.rename(columns={('cases', '0-9'): 'Cases_0_9',
                                ('cases', '10-19'): 'Cases_10_19',
                                ('cases', '20-29'): 'Cases_20_29',
                                ('cases', '30-39'): 'Cases_30_39',
                                ('cases', '40-49'): 'Cases_40_49',
                                ('cases', '50-59'): 'Cases_50_59',
                                ('cases', '60-69'): 'Cases_60_69',
                                ('cases', '70-79'): 'Cases_70_79',
                                ('cases', '80-89'): 'Cases_80_89',
                                ('cases', '90+'): 'Cases_90_Plus',
                                'IC_Bedden_COVID': 'ICU_Beds',
                                'Kliniek_Bedden': 'Hosp_Beds',
                                'IC_Nieuwe_Opnames_COVID': 'ICU_Inflow',
                                'Kliniek_Nieuwe_Opnames_COVID': 'Hosp_Inflow',
                                'Totaal_bezetting': 'Total_Beds',
                                'IC_Opnames_7d': 'ICU_Inflow_SMA7d',
                                'Kliniek_Opnames_7d': 'Hosp_Inflow_SMA7d',
                                'Totaal_opnames': 'Total_Inflow',
                                'Totaal_opnames_7d': 'Total_Inflow_SMA7d',
                                'Totaal_IC': 'Total_ICU_Beds',
                                'IC_opnames_14d': 'ICU_Inflow_SMA14d',
                                'Kliniek_opnames_14d': 'Hosp_Inflow_SMA14d',
                                'OMT_Check_IC': 'OMT_Check_ICU',
                                'OMT_Check_Kliniek': 'OMT_Check_Hosp',
                                'prev_low': 'Prev_LB',
                                'prev_avg': 'Prev',
                                'prev_up': 'Prev_UB',
                                'Rt_low': 'R_LB',
                                'Rt_avg': 'R',
                                'Rt_up': 'R_UB',
                                'values.tested_total': 'Tested',
                                'values.infected': 'Cases',
                                'values.infected_percentage': 'Cases_Pct',
                                'tests.7d.avg': 'Tested_SMA7d',
                                'pos.rate.3d.avg': 'Cases_Pct_SMA3d',
                                'pos.rate.7d.avg': 'Cases_Pct_SMA7d',
                                'vaccines_administered_estimated_carehomes': 'Vacc_Est_Carehomes',
                                'vaccines_administered_ggd': 'Vacc_Adm_GGD',
                                'vaccines_administered_hospital': 'Vacc_Adm_Hosp',
                                'vaccines_administered_estimated': 'Vacc_Est',
                                'vaccines_administered': 'Vacc_Adm',
                                'vaccines_administered_doctors': 'Vacc_Adm_Doctors',
                                'RNA_flow_per_100000': 'RNA_Sum',
                                'RNA_per_100000_per_measurement': 'RNA',
                                'pandas_SMA_3': 'RNA_SMA3d',
                                'pandas_SMA_7': 'RNA_SMA7d'
                                })



### DEALING WITH UNAVAILABLE DATA

# Replace vaccine variable nan by 0 since vaccines only start at 2021-01-06
master[['Vacc_Est_Carehomes', 'Vacc_Adm_GGD', 'Vacc_Adm_Hosp', 'Vacc_Est',
       'Vacc_Adm', 'Vacc_Adm_Doctors']] = master[['Vacc_Est_Carehomes',
                                                  'Vacc_Adm_GGD',
                                                  'Vacc_Adm_Hosp',
                                                  'Vacc_Est',
                                                  'Vacc_Adm',
                                                  'Vacc_Adm_Doctors']].fillna(0)

# replace date where vaccin columns are all zero with nan
master.loc[['2021-04-11'], ['Vacc_Est', 'Vacc_Est_Carehomes', 'Vacc_Adm_GGD',
            'Vacc_Adm_Hosp', 'Vacc_Adm_Doctors']] = np.NAN

# Interpolate 2021-02-07 Cases and Cases_Pct due to Code Red (extreme weather)
master.loc[['2021-02-07'], ['Cases', 'Cases_Pct']] = np.NAN

# Interpolate nans in RNA data and in vaccin columns
master = master.interpolate()


### Defining additional variables

# Calculate 3-day and 7-day SMA of several variables
master['ICU_Inflow_SMA3d'] = master.ICU_Inflow.rolling(window=3).mean()
master['Hosp_Inflow_SMA3d'] = master.Hosp_Inflow.rolling(window=3).mean()
master['Tested_SMA3d'] = master.Tested.rolling(window=3).mean()
master['Cases_SMA3d'] = master.Cases.rolling(window=3).mean()
master['Cases_SMA7d'] = master.Cases.rolling(window=7).mean()
master['Cases_Pct_SMA3d'] = master.Cases_Pct.rolling(window=3).mean()
master['Cases_Pct_SMA7d'] = master.Cases_Pct.rolling(window=7).mean()
master['Vacc_Est_SMA3d'] = master.Vacc_Est.rolling(window=3).mean()
master['Vacc_Est_SMA7d'] = master.Vacc_Est.rolling(window=7).mean()


# Calculate R and Prevalence as average between LB and UB
master.R = (master.R_LB + master.R_UB)/2
master.Prev = (master.Prev_LB + master.Prev_UB)/2

# Delay R, Prevalence and RNA by 7 days to get values at recent days
master.R = master.R.shift(7)
master.Prev = master.Prev.shift(7)
master.RNA = master.RNA.shift(7)
master.RNA_SMA3d = master.RNA_SMA3d.shift(7)
master.RNA_SMA7d = master.RNA_SMA7d.shift(7)

# Remove all data up to and including 29-10-2020 as SMA variables contain nan
# This is a temporary solution. We might later decide to exclude these variables
# to use the observations before 29-10-2020 as well
master = master.loc['2020-10-30':]

# We downloaded the data on 2021-05-05.
# We remove all data after 2021-05-02 since many variables are unavailable. This is because we use open source data.
# When our model is employed by the LCPS all relevant data is available.
master = master.loc[:'2021-05-02']

# Add day of the week dummies
names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, x in enumerate(names):
    master[x] = (pd.to_datetime(master.index, format='%Y-%m-%d').get_level_values(0).weekday == i).astype(int)


### SAVING

# Copy master dataframe to other dataframe for saving
all_data = master.copy()

# Create subset of all data with relevant variables.
rel_vars = ['ICU_Inflow', 'ICU_Inflow_SMA3d',
            'ICU_Inflow_SMA7d', 'ICU_Inflow_SMA14d',
            'Hosp_Inflow', 'Hosp_Inflow_SMA3d',
            'Hosp_Inflow_SMA7d', 'Hosp_Inflow_SMA14d',
            'Tested', 'Tested_SMA3d', 'Tested_SMA7d',
            'Cases', 'Cases_SMA3d', 'Cases_SMA7d',
            'Cases_Pct', 'Cases_Pct_SMA3d', 'Cases_Pct_SMA7d',
            'Prev', 'R', 'RNA', 'RNA_SMA3d', 'RNA_SMA7d',
            'Vacc_Est', 'Vacc_Est_SMA3d', 'Vacc_Est_SMA7d',
            'Vacc_Est_Carehomes', 'Vacc_Adm_GGD',
            'Vacc_Adm_Hosp', 'Vacc_Adm_Doctors',
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']

master = master[rel_vars]

# Save dataframe with all data to file 'all_data'
all_data.to_csv(r'Data\all_data.csv')

# Save dataframe with relevant variables to file 'master
master.to_csv(r'Data\master.csv')




