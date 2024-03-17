"""
COMP247-402
Final Project: Phase 1

File name:     Group4_Project_Phase1.py
Student names: Damien Liscio
               Cole Ramsey
               Victor Zorn
               Garnett Grant
Due date:      Mar. 17, 2024
"""

# imports
import pandas as pd
import matplotlib.pyplot as plt

# Section 1: Data exploration

# load KSI dataset into data frame
KSI_data = pd.read_csv('KSI.csv')

# basic column exploration
column_names = KSI_data.columns # get column names
column_types = KSI_data.dtypes # get column types
column_missing_data = KSI_data.isnull().sum() # get sum of missing values for each column
column_descriptions = KSI_data.describe() # get column descriptions

# print name, number of missing values, and data type for each column
print(KSI_data.info())

# loop through each column
for column in column_names:
    # check if column is in list of columns that have descriptions
    column_description = '\nColumn Description'
    if column in column_descriptions:
        column_description += ':\n' + str(column_descriptions[column])
    else:
        column_description += ' Not Available'
    
    # display column names, types and descriptions (if applicable)
    print('Column Name:', column, 'Column Type:', column_types[column],
          'Sum Of Column Missing Values:', column_missing_data[column],
          column_description, '\n')

# plot quantity of each injury type by year
injury_type_counts = KSI_data.groupby(['YEAR', 'INJURY']).size().unstack(fill_value=0) # get quantities for injury counts by year
injury_type_counts.plot(kind='bar', stacked=True, figsize=(8,6)) # plot data as bar graph
plt.xlabel('Year')
plt.ylabel('Number of Reports Per Injury Type')
plt.title('Quantity of Injury Type by Year')
plt.legend(title='Injury Type')
plt.show()
