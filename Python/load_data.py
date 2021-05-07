import pandas as pd

################ LOAD DATA AND COMBINE DATA INTO MASTER DATAFRAME #######################3
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
data_vaccines = pd.read_csv("Data/vaccines.csv",
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
#   There is only data every 3 weeks until april 20th available, so not the best source of data


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

# Rename variables in all datasets to make code foolproof and consistent
# Put code here (note this can also be done before transforming date column, but then that code needs to be adjusted)


# Delete superfluous variables, such as data_rivm_reproduction.population and data_rivm_prevalence.version
# 1. Variables below have the same value over all observations
data_rivm_prevalence = data_rivm_prevalence.drop(labels=["population", "version"], axis=1)
data_rivm_reproduction = data_rivm_reproduction.drop(labels="version", axis=1)


# Only keep data from 2020-10-17 (start LCPS IC admission dataset) until 2021-05-02 (end RIVM prevalence dataset)
# 1. Make the date column the index of the dataset.
# This has already been done for agg_ggd_tests_wide
data_lcps_admissions = data_lcps_admissions.set_index("date")
data_rivm_prevalence = data_rivm_prevalence.set_index("date")
data_rivm_reproduction = data_rivm_reproduction.set_index("date")
data_rivm_tests = data_rivm_tests.set_index("date")
data_vaccines = data_vaccines.set_index("date")
data_rna = data_rna.set_index("date")

# 2. Slice all dates from now to 2020-10-17
data_lcps_admissions = data_lcps_admissions.loc[:'2020-10-17']
data_rivm_prevalence = data_rivm_prevalence.loc[:'2020-10-17']
data_rivm_reproduction = data_rivm_reproduction.loc[:'2020-10-17']
data_rivm_tests = data_rivm_tests.loc[:'2020-10-17']
data_vaccines = data_vaccines.loc[:'2020-10-17']
agg_ggd_tests_wide = agg_ggd_tests_wide.loc[:'2020-10-17']
data_rna = data_rna.loc[:'2020-10-17']

# Load aggregated RNA data created in rna_data.py
# Note that the file RNA_graphs.py uses mzelst.csv as a data source.
# This needs to be changed since mzelst.csv is no longer used
# Put code here


# Create master dataframe containing all datasets
master = pd.concat([agg_ggd_tests_wide, data_lcps_admissions,
                    data_rivm_prevalence, data_rivm_reproduction,
                    data_rivm_tests, data_vaccines, data_rna], axis=1)

# dataset is niet gesort.
# vaccines beginnen pas vanaf 2021-01-06

# Save master dataframe to file 'master_data'
master.to_csv(r'Data\master.csv')

