"""
COMP247-402
Final Project: Phase 1

File name:     Group4_Project_Phase1.py
Student names: Garnett Grant
               Damien Liscio
               Cole Ramsey
               Avalon Stanley
               Victor Zorn
Due date:      Mar. 17, 2024
"""

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split

# Section 1: Data exploration

# load KSI dataset into data frame
KSI_data = pd.read_csv('KSI.csv')

# preliminary data exploration
print(KSI_data.head())

# print name, number of missing values, and data type for each column
print(KSI_data.info())

# display column description only if column has description
column_descriptions = KSI_data.describe()
for column in column_descriptions.columns:
    print(column_descriptions[column], '\n')

# plot quantity of each injury type by year as bar graph
injury_type_counts = KSI_data.groupby(['YEAR', 'INJURY']).size().unstack(fill_value=0) # get quantities for injury counts by year
injury_type_counts.plot(kind='bar', stacked=True, figsize=(8,6))
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

# plotting number of road accidents by impact type for each injury category
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

# plotting number of road accidents for each hour of the day
# convert time to string and fill with 0's if necessary
KSI_data['TIME'] = KSI_data['TIME'].astype(str).str.zfill(4)
# create function to return appropriate time range for accident
def time_ranges(time):
    hour = int(time[:2])
    hour_range_start = hour // 1 * 1  
    return f'{hour_range_start:02d}:00-{hour_range_start:02d}:59'
# create new column with ranges as values
KSI_data['TimeRange'] = KSI_data['TIME'].apply(time_ranges)
# get counts for each range and plot 
time_range_counts = KSI_data.groupby('TimeRange').size()
time_range_counts.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Time (1 hr intervals)')
plt.ylabel('Number of Road Accidents')
plt.title('Number of Accidents by Hour in 24 Hour Period')
plt.xticks(rotation=45)  
plt.tight_layout() 
plt.show()

# Section 2: Data modelling

# preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# bundling preprocessing for num and cat data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ]
)

# dropping rows where ACCLASS data is NaN
KSI_data.dropna(subset=['ACCLASS'], inplace=True)

# splitting dataset into training and testing sets
X = KSI_data.drop(['ACCLASS'], axis=1)
y = KSI_data['ACCLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.head())
print(y_train.head())
