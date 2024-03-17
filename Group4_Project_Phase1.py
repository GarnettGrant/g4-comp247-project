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
import seaborn as sns

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

# calculating the correlations of data 
corr = KSI_data.select_dtypes(include=['float64', 'int64']).corr()
# plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# plotting the frequency of 'ALCOHOL' involvement in fatal accidents
# replacing NaN values in ALCOHOL column with "No"
KSI_data['ALCOHOL'] = KSI_data['ALCOHOL'].fillna('No')
print(KSI_data['ALCOHOL'].value_counts())
fatal_accidents = KSI_data[KSI_data['ACCLASS'] == 'Fatal']
plt.figure(figsize=(8, 6))
sns.countplot(data=fatal_accidents, x='ALCOHOL')
plt.title('Frequency of Alcohol Involvement in Fatal Accidents')
plt.xlabel('Alcohol Involvement')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# plotting numbner of road accidents by impact type for each injury category.
accident_count = KSI_data.groupby(['IMPACTYPE',  'INJURY']).size().unstack(fill_value=0) 
accident_count = accident_count.reset_index()
accident_count.plot(kind='bar', stacked=True, figsize=(12,8))
plt.xticks(range(len(accident_count['IMPACTYPE'])), accident_count['IMPACTYPE'], rotation=30)
plt.xlabel('Impact Type')
plt.ylabel('Number of Accidents')
plt.title('Quantity of Injury Type by Impact Type')
plt.legend(title='Impact Type')
plt.show()

# plotting number of road accidents given the road class and location ordinants
accident_counts = KSI_data.groupby(['ROAD_CLASS', 'LOCCOORD']).size().unstack(fill_value=0)
accident_counts.plot(kind='bar', stacked=True, figsize=(8,6))
plt.xlabel('Road Class')
plt.ylabel('Number of Reports')
plt.title('Number of Accidents by Road Class and Location Coordinates')
plt.xticks(rotation=20)
plt.legend(title='Location Coordinates')
plt.show()