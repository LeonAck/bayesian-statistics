import pandas as pd

# Load data
data_mzelst = pd.read_csv("Data/mzelst.csv")
data_lcps_ic = pd.read_csv("Data/lcps_ic_opnames.csv")
data_nice_ic = pd.read_csv("Data/nice_ic_intake.csv")
data_rivm_ic = pd.read_csv("Data/rivm_ic_opnames.csv", sep=';')
data_rivm_riool = pd.read_csv("Data/rivm_rioolwaterdata.csv", sep=';')

# Inspect variable types
data_mzelst.info()
data_lcps_ic.info()
data_nice_ic.info()
data_rivm_ic.info()

# Transform date column from string to datetime
data_mzelst.date = pd.to_datetime(data_mzelst.date, format='%Y-%m-%d')
data_lcps_ic.Datum = pd.to_datetime(data_lcps_ic.Datum, format='%d-%m-%Y')
data_nice_ic.date = pd.to_datetime(data_nice_ic.date, format='%Y-%m-%d')
data_rivm_ic.Date_of_statistics = pd.to_datetime(data_rivm_ic.Date_of_statistics, format='%Y-%m-%d')
data_rivm_riool.Date_measurement =

# Sort dataframes to get newest date first
data_mzelst = data_mzelst.sort_values(by='date', ascending=False)
data_lcps_ic = data_lcps_ic.sort_values(by='Datum', ascending=False)
data_nice_ic = data_nice_ic.sort_values(by='date', ascending=False)
data_rivm_ic = data_rivm_ic.sort_values(by='Date_of_statistics', ascending=False)

# Delete the newest day since it is not complete
data_mzelst = data_mzelst.drop(427, axis=0)
data_lcps_ic = data_lcps_ic.drop(0, axis=0)
data_nice_ic = data_nice_ic.drop(427, axis=0)

# Inspect differences in IC admission data among data sources
data_lcps_ic.IC_Nieuwe_Opnames_COVID.head(10) #Het aantal patiënten met COVID-19 dat in 24 uur nieuw is opgenomen op de IC
data_nice_ic.IC_Intake.head(10) #Aantal nieuwe patiënten met verdachte of bewezen COVID-19 status die per dag op de IC’s worden opgenomen
data_rivm_ic.IC_admission.head(10) #RIVM haalt z'n data van het NICE, waarom het (lichtelijk) afwijkt is mij onbekend
#De IC admission data van het LCPS lijkt een dag achter te lopen op die van het RIVM/NICE, waarom is mij onbekend
#De IC admission data van het LCPS is consistent lager dan die van het RIVM/NICE, waarom is mij onbekend






