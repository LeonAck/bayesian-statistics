import pandas as pd

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
# until 2021-05-02 (end RIVM prevalence dataset)
# 1. Make the date column the index of the dataset.
# This has already been done for agg_ggd_tests_wide
data_lcps_admissions = data_lcps_admissions.set_index("date")
data_rivm_prevalence = data_rivm_prevalence.set_index("date")
data_rivm_reproduction = data_rivm_reproduction.set_index("date")
data_rivm_tests = data_rivm_tests.set_index("date")
data_vaccines = data_vaccines.set_index("date")
data_rna = data_rna.set_index("date")

# 2. Slice all dates from 2021-05-02 to 2020-10-17
data_lcps_admissions = data_lcps_admissions.loc['2021-05-02':'2020-10-17']
data_rivm_prevalence = data_rivm_prevalence.loc['2021-05-02':'2020-10-17']
data_rivm_reproduction = data_rivm_reproduction.loc['2021-05-02':'2020-10-17']
data_rivm_tests = data_rivm_tests.loc['2021-05-02':'2020-10-17']
data_vaccines = data_vaccines.loc['2021-05-02':'2020-10-17']
agg_ggd_tests_wide = agg_ggd_tests_wide.loc['2021-05-02':'2020-10-17']
data_rna = data_rna.loc['2021-05-02':'2020-10-17']


# Create master dataframe containing all datasets
master = pd.concat([agg_ggd_tests_wide, data_lcps_admissions,
                    data_rivm_prevalence, data_rivm_reproduction,
                    data_rivm_tests, data_vaccines, data_rna], axis=1)

# Sort master dataframe to get newest data first
master = master.sort_values(by='date', ascending=False)

# Delete superfluous variables
master = master.drop(labels=[('cases', '<50'), ('cases', 'Unknown'),
                             'population', 'RNA_per_ml', 'RNA_flow_per_100000',
                             'Measurement_count'], axis=1)

# Rename variables for clarity and consistency
master = master.rename(columns={('cases', '0-9'): 'Cases_0-9',
                                ('cases', '10-19'): 'Cases_10-19',
                                ('cases', '20-29'): 'Cases_20-29',
                                ('cases', '30-39'): 'Cases_30-39',
                                ('cases', '40-49'): 'Cases_40-49',
                                ('cases', '50-59'): 'Cases_50-59',
                                ('cases', '60-69'): 'Cases_60-69',
                                ('cases', '70-79'): 'Cases_70-79',
                                ('cases', '80-89'): 'Cases_80-89',
                                ('cases', '90+'): 'Cases_90+',
                                'IC_Bedden_COVID': 'ICU_Beds_COVID',
                                'IC_Bedden_Non_COVID': 'ICU_Beds_Non_COVID',
                                'Kliniek_Bedden': 'Hosp_Beds_COVID',
                                'IC_Nieuwe_Opnames_COVID': 'ICU_Inflow_COVID',
                                'Kliniek_Nieuwe_Opnames_COVID': 'Hosp_Inflow_COVID',
                                'Totaal_bezetting': 'Total_Beds_COVID',
                                'IC_Opnames_7d': 'ICU_Inflow_COVID_SMA7d',
                                'Kliniek_Opnames_7d': 'Hosp_Inflow_COVID_SMA7d',
                                'Totaal_opnames': 'Total_Inflow_COVID',
                                'Totaal_opnames_7d': 'Total_Inflow_COVID_SMA7d',
                                'Totaal_IC': 'Total_ICU_Beds',
                                'IC_opnames_14d': 'ICU_Inflow_COVID_SMA14d',
                                'Kliniek_opnames_14d': 'Hosp_Inflow_COVID_SMA14d',
                                'OMT_Check_IC': 'OMT_Check_ICU',
                                'OMT_Check_Kliniek': 'OMT_Check_Hosp',
                                'prev_low': 'Prev_LB',
                                'prev_avg': 'Prev',
                                'prev_up': 'Prev_UB',
                                'groei_besmettelijken': 'Prev_Growth',
                                'besmet_7daverage': 'Prev_SMA7d',
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


# Replace vaccine variable nan by 0 since vaccines only start at 2021-01-06
master[['Vacc_Est_Carehomes', 'Vacc_Adm_GGD', 'Vacc_Adm_Hosp', 'Vacc_Est',
       'Vacc_Adm', 'Vacc_Adm_Doctors']] = master[['Vacc_Est_Carehomes',
                                                  'Vacc_Adm_GGD',
                                                  'Vacc_Adm_Hosp',
                                                  'Vacc_Est',
                                                  'Vacc_Adm',
                                                  'Vacc_Adm_Doctors']].fillna(0)


# Save master dataframe to file 'master_data'
master.to_csv(r'Data\master.csv')
