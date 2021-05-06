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
#data_vaccines = pd.read_csv("Data/vaccines.csv")
#   The vaccine dataset does not seem to load properly, I do not know why
#data_rivm_gedrag = pd.read_csv("Data/rivm_gedrag.csv", sep=';')
#   Preparing the gedrag dataset is on hold due to time efficiency
#   There is only data every 3 weeks until april 20th available, so not the best source of data


# Transform date column from string to datetime
data_ggd_tests.date = pd.to_datetime(data_ggd_tests.date, format='%Y-%m-%d')
data_lcps_admissions.date = pd.to_datetime(data_lcps_admissions.date, format='%Y-%m-%d')
data_rivm_prevalence.Date = pd.to_datetime(data_rivm_prevalence.Date, format='%Y-%m-%d')
data_rivm_reproduction.Date = pd.to_datetime(data_rivm_reproduction.Date, format='%Y-%m-%d')
data_rivm_tests.date = pd.to_datetime(data_rivm_tests.date, format='%Y-%m-%d')


# Sort dataframes to get newest date first
# Put code here


# Rename variables in all datasets to make code foolproof and consistent
# Put code here (note this can also be done before transforming date column, but then that code needs to be adjusted)


# Delete superfluous variables, such as data_rivm_reproduction.population and data_rivm_prevalence.version
# Put code here


# Only keep data from 2020-10-17 (start LCPS IC admission dataset) until 2021-05-02 (end RIVM prevalence dataset)
# Put code here


# GGD dataset: aggregate data per GGD on a national level per age group
# Also transform from long into wide format (i.e. with only date as a row index, and cases_0-9 cases_10-19 etc as column variables)
# Put code here


# Load aggregated RNA data created in rna_data.py
# Put code here


# Create master dataframe containing all datasets
# Put code here


# Save master dataframe to file 'master_data'
# Put code here






