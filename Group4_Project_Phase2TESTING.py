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
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

# Section 1: Data exploration

# load KSI dataset into data frame
KSI_data = pd.read_csv("C:/Users/coles/Downloads/KSI.csv")

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

# calculating correlations
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
accident_count = KSI_data.groupby(['IMPACTYPE', 'INJURY']).size().unstack(fill_value=0) 
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



############
# Section 3: Predictive model building
############


#################
# logistic regression
# model pipeline
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', MaxAbsScaler()),
    ('classifier', LogisticRegression(solver='saga', max_iter=5000))
])

# randomizedsearchcv params
param_dist_lr = {
    'classifier__C': loguniform(1e-4, 10)
}

# gridsearchcv params
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__max_iter': [2000, 5000, 10000]
}

# setting up grid search
grid_search_lr = GridSearchCV(estimator=pipeline_lr, param_grid=param_grid_lr, cv=5, scoring='accuracy', verbose=1)

# setting up random grid search
random_search_lr = RandomizedSearchCV(estimator=pipeline_lr, param_distributions=param_dist_lr, n_iter=20, cv=5, scoring='accuracy', verbose=1, random_state=42)

# fitting grid search
grid_search_lr.fit(X_train, y_train)

# fitting random grid search
random_search_lr.fit(X_train, y_train)

# best params and best score grid search LOGISTIC REGRESSION
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best score for Logistic Regression:", grid_search_lr.best_score_)

# best params and best score randomized grid search LOGISTIC REGRESSION
print("Best parameters for Logistic Regression (Randomized):", random_search_lr.best_params_)
print("Best score for Logistic Regression (Randomized):", random_search_lr.best_score_)



################
# decision trees
# model pipeline
pipeline_dt = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# gridsearchcv params
param_grid_dt = {
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 10, 20],
    'classifier__min_samples_leaf': [1, 5, 10]
}

# randomizedsearchcv params
param_dist_dt = {
    'classifier__max_depth': randint(5, 50),
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10)
}

# setting up grid search
grid_search_dt = GridSearchCV(estimator=pipeline_dt, param_grid=param_grid_dt, cv=5, scoring='accuracy', verbose=1)

# setting up random grid search
random_search_dt = RandomizedSearchCV(estimator=pipeline_dt, param_distributions=param_dist_dt, n_iter=20, cv=5, scoring='accuracy', verbose=1, random_state=42)

# fitting grid search
grid_search_dt.fit(X_train, y_train)

# fitting random grid search
random_search_dt.fit(X_train, y_train)

# best params and best score DECISION TREE
print("Best parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best score for Decision Tree:", grid_search_dt.best_score_)

# best params and best score randomized grid search DECISION TREE
print("Best parameters for Decision Tree (Randomized):", random_search_dt.best_params_)
print("Best score for Decision Tree (Randomized):", random_search_dt.best_score_)




################
# SVM
# model pipeline
pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42, probability=True))
])

# gridsearchcv params
param_grid_svm = {
    'classifier__C': [1, 10],
    'classifier__kernel': ['rbf']
}

# randomizedsearchcv params
param_dist_svm = {
    'classifier__C': loguniform(1e-1, 10),
    'classifier__gamma': loguniform(1e-4, 1e-1),
    'classifier__kernel': ['rbf']
}

# setting up grid search
grid_search_svm = GridSearchCV(estimator=pipeline_svm, param_grid=param_grid_svm, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

# setting up random grid search
random_search_svm = RandomizedSearchCV(estimator=pipeline_svm, param_distributions=param_dist_svm, n_iter=10, cv=3, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)

# fitting grid search
grid_search_svm.fit(X_train, y_train)

# fitting random grid search
random_search_svm.fit(X_train, y_train)

# best params and best score SVM
print("Best parameters for SVM:", grid_search_svm.best_params_)
print("Best score for SVM:", grid_search_svm.best_score_)

# best params and best score randomized grid search SVM
print("Best parameters for SVM (Randomized):", random_search_svm.best_params_)
print("Best score for SVM (Randomized):", random_search_svm.best_score_)



################
# random forest
# model pipeline
pipeline_rf_grid = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# gridsearchcv params
param_grid_rf = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [10, 20],
    'classifier__min_samples_leaf': [5, 10]
}

# randomizedsearchcv params
param_dist_rf = {
    'classifier__n_estimators': randint(50, 200),
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10)
}

# setting up grid search
grid_search_rf = GridSearchCV(estimator=pipeline_rf_grid, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=1)

# setting up random grid search
random_search_rf = RandomizedSearchCV(estimator=pipeline_rf_grid, param_distributions=param_dist_rf, n_iter=20, cv=5, scoring='accuracy', verbose=1, random_state=42)

# fitting grid search
grid_search_rf.fit(X_train, y_train)

# fitting random grid search
random_search_rf.fit(X_train, y_train)

# best params and best score RANDOM FOREST
print("Best parameters for Random Forest (Grid Search):", grid_search_rf.best_params_)
print("Best score for Random Forest (Grid Search):", grid_search_rf.best_score_)

# best params and best score randomized grid search RANDOM FOREST
print("Best parameters for Random Forest (Randomized):", random_search_rf.best_params_)
print("Best score for Random Forest (Randomized):", random_search_rf.best_score_)




################
# neural networks
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# model pipeline
pipeline_nn = Pipeline([
    ('preprocessor', preprocessor),
    ('svd', TruncatedSVD(n_components=50)),
    ('classifier', MLPClassifier(max_iter=100, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5))
])

# gridsearchcv params
param_grid_nn = {
    'classifier__hidden_layer_sizes': [(50,)],
    'classifier__activation': ['tanh'],
    'classifier__alpha': [0.0001],
    'classifier__learning_rate_init': [0.001]
}

# randomizedsearchcv params
param_dist_nn = {
    'classifier__hidden_layer_sizes': [(50,), (100,)],
    'classifier__activation': ['tanh', 'relu'],
    'classifier__alpha': loguniform(1e-4, 1e-2),
    'classifier__learning_rate_init': loguniform(1e-4, 1e-2)
}

# setting up grid search
grid_search_nn = GridSearchCV(estimator=pipeline_nn, param_grid=param_grid_nn, cv=3, scoring='accuracy', verbose=1)

# setting up random grid search
random_search_nn = RandomizedSearchCV(estimator=pipeline_nn, param_distributions=param_dist_nn, n_iter=10, cv=3, scoring='accuracy', verbose=1, random_state=42)

# fitting grid search
grid_search_nn.fit(X_train, y_train)

# fitting random grid search
random_search_nn.fit(X_train, y_train)

# best params and best score NEURAL NETWORK
print("Best parameters for Neural Network (Grid Search):", grid_search_nn.best_params_)
print("Best score for Neural Network (Grid Search):", grid_search_nn.best_score_)

# best params and best score randomized grid search NEURAL NETWORK
print("Best parameters for Neural Network (Randomized):", random_search_nn.best_params_)
print("Best score for Neural Network (Randomized):", random_search_nn.best_score_)



############
# Section 4: Model scoring and evaluation
############

def evaluate_model(name, model, features, labels):
    pred = model.predict(features)
    acc = accuracy_score(labels, pred)
    print(f'Accuracy of {name}: {acc:.2f}')
    
    #confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(labels, pred))
    
    #classification report
    print("Classification Report:")
    print(classification_report(labels, pred, zero_division=1))
    
    #computing ROC curve and ROC area for each class
    pred_prob = model.predict_proba(features)[:, 1]
    fpr, tpr, _ = roc_curve(labels, pred_prob, pos_label=model.classes_[1])
    roc_auc = auc(fpr, tpr)
    
    #plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic of {name}')
    plt.legend(loc="lower right")
    plt.show()
    
#evaluating each model
evaluate_model('Logistic Regression', grid_search_lr.best_estimator_, X_test, y_test)
evaluate_model('Decision Tree', grid_search_dt.best_estimator_, X_test, y_test)
evaluate_model('SVM', grid_search_svm.best_estimator_, X_test, y_test)
evaluate_model('Random Forest', grid_search_rf.best_estimator_, X_test, y_test)
evaluate_model('Neural Network', grid_search_nn.best_estimator_, X_test, y_test)   
