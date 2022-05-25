import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error

'''------------------------------------------------------------------'''
# import train data
train_data = pd.read_csv('/Users/cman/Desktop/code/titanic/train.csv')
# print(train_data.columns)

# import test data
test_data = pd.read_csv('/Users/cman/Desktop/code/titanic/test.csv')
# print(test_data.columns)

# test + train
dfs = [train_data, test_data]
# print(train_data.columns)
# print(test_data.columns)

# test + train DataFrame
all_data = pd.concat([train_data, test_data]).reset_index(drop=True)

# naming each data set
train_data.name = 'Training Data'
test_data.name = 'Testing Data'
all_data.name = 'All Data'

'''------------------------------------------------------------------'''
# def function to show NaN values
'''def nan_val(df):
	for col in df.columns:
		if df[col].isnull().sum() != 0:
			print('{} column has {} missing data points'.format(col, df[col].isnull().sum()))
	print('\n')

for df in dfs:
	print('{}'.format(df.name))
	nan_val(df)'''

'''------------------------------------------------------------------'''

# fix 177 missing age NaN's in training data and testing data
corr_matrix = train_data.corr().abs()
corr_matrix_ = corr_matrix.unstack()

# sorting correlation matrix
corr_matrix_sort = corr_matrix_.sort_values(kind='quicksort', ascending=False).reset_index()

# creating new descriptive columns from sorted correlation
corr_matrix_resort = corr_matrix_sort.rename(columns={'level_0':'feature 1', 'level_1':'feature 2', 0:'corr'})

# which feature is most correlated to age?
print(corr_matrix_resort[corr_matrix_resort['feature 1'] == 'Age'])

'''------------------------------------------------------------------'''

# heatmap of feature correlations


plt.figure(figsize = (8,6))
sns.heatmap(corr_matrix, annot=True, cbar=True, linewidths=0.3, linecolor='black')
plt.title('Feature correlation', fontsize=15)
plt.xlabel('Feature 1', labelpad=-18)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Feature 2', labelpad=-5)
plt.yticks(rotation=45, fontsize=10)
plt.show()


'''------------------------------------------------------------------'''

# fix 687 missing cabin NaN's in training data and 1 in testing data
age_by_pclass = all_data.groupby(['Pclass']).median()['Age']

for pclass in range(1, 4):
	print('Median age of Pclass {} is: {}'.format(pclass, age_by_pclass[pclass]))


# fix 2 missing embarked NaN's in training data and 327 in testing data



'''------------------------------------------------------------------'''








'''------------------------------------------------------------------'''
# women survival rate
women = train_data.loc[train_data.Sex == 'female']['Survived']
women_rate = sum(women)/len(women)
# print('The % of women that survived is:', women_rate*100)

# men survival rate
men = train_data.loc[train_data.Sex == 'male']['Survived']
men_rate = sum(men)/len(men)
# print('The % of men that survived is:', men_rate*100)

'''------------------------------------------------------------------'''

# class 1 survival rate
pclass1 = train_data.loc[train_data.Pclass == 1]['Survived']
rate_pclass1 = sum(pclass1)/len(pclass1)
# print('The % of First Class that survived is:', rate_pclass1*100)

# class 2 survival rate
pclass2 = train_data.loc[train_data.Pclass == 2]['Survived']
rate_pclass2 = sum(pclass2)/len(pclass2)
# print('The % of Second Class that survived is:', rate_pclass2*100)

# class 3 survival rate
pclass3 = train_data.loc[train_data.Pclass == 3]['Survived']
rate_pclass3 = sum(pclass3)/len(pclass3)
# print('The % of Third Class that survived is:', rate_pclass3*100)

'''------------------------------------------------------------------'''
# sikit random forest 
# y variable
y = train_data['Survived']


# variables looking into
features = ['Pclass', 'Sex', 'SibSp', 'Parch']

# indicator train variables
X = pd.get_dummies(train_data[features])

# indicator test variables
X_test = pd.get_dummies(test_data[features])

# model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# fitting the model
fit_model = model.fit(X, y)

# prediction
prediction = fit_model.predict(X_test)

output_pred = pd.DataFrame({'Survived': prediction})

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)

tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
	print('Scores:', scores)
	print('Mean:', scores.mean())
	print('Standard deviation:', scores.std())


# print(display_scores(tree_rmse_scores))















