import pandas as pd
import matplotlib.pyplot as plt

KSI_data = pd.read_csv("/Users/damie/Downloads/KSI.csv")  #Load KSI dataset into data frame

#Basic column exploration
column_names = KSI_data.columns    #Get column names
column_types = KSI_data.dtypes   #Get column types
column_missing_data = KSI_data.isnull().sum()  #Get sum of missing values for each column
column_descriptions = KSI_data.describe()  #Get column descriptions
for column in column_names:    #Loop through each column
    if column in column_descriptions:    #Check if the column is in list of columns that have descriptions
        print('Column Name:',column, 'Column Type:',column_types[column], 'Sum Of Column Missing Values:', column_missing_data[column], '\nColumn Description:\n',column_descriptions[column], '\n')    #Display column names, types and descriptions
    else:
        print('Column Name:',column, 'Column Type:',column_types[column], 'Sum Of Column Missing Values:', column_missing_data[column], '\nColumn Description Not Available\n')  #Display column names and types for those without descriptions
        
#Plot quantity of each injury type by year
injury_type_counts = KSI_data.groupby(['YEAR', 'INJURY']).size().unstack(fill_value=0)  #Get quantities for injury counts by year
plt.figure(figsize=(8,6))  #Set plot size
injury_type_counts.plot(kind='bar', stacked=True)   #Plot data as bar graph
plt.xlabel('Year')          #Set plot xlabel
plt.ylabel('Number of Reports Per Injury Type')   #Set plot ylabel
plt.title('Quantity of Injury Type by Year')   #set plot title
plt.legend(title='Injury Type')   #Set plot legends title
plt.show()   #Show the plot



