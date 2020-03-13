import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

df_math = pd.read_csv("student-mat.csv", sep=";", names=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                                                         'Medu',
                                                         'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime',
                                                         'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                                                         'activities', 'nursery', 'higher', 'internet', 'romantic',
                                                         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                                                         'absences', 'G1', 'G2', 'G3'])

# data exploration
maths = df_math.drop(index=0, axis=0)
print()
print('the data for the people offering maths')
print(maths.head())

# maths
print('the info from maths', '\n', maths.info())
print('the description from maths', '\n', maths.describe())

print(maths.isnull().sum().sort_values(ascending=False))
print()
for cols in maths.columns:
    pass
#    print(maths[cols].value_counts())
#    print()
# print('Checking for missing values')
# print(maths.isnull().sum().sort_values())

# print(maths.columns)
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
        'G1', 'G2', 'G3']

# for i in cols:
#    plt.figure(figsize=(8, 8))
#    sns.countplot(maths[i], palette=sns.color_palette("cubehelix"))
#    plt.show()

df = maths.copy()
int_features = ['age', 'Fedu', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1',
                'G2', 'G3', 'freetime', 'goout', 'health', 'Medu', 'studytime', 'traveltime']
print(len(int_features))

for vals in int_features:
    df[vals] = df[vals].astype(int)

print('the number of numerical features we have in this dataset is set at', df.info())

# df['G1'] = df['G1'].astype(int)
# df['G2'] = df['G2'].astype(int)
# df['G3'] = df['G3'].astype(int)
# df['age'] = df['age'].astype(int)


score_map = {0: 'Below Average', 1: 'Below Average', 2: 'Below Average', 3: 'Below Average', 4: 'Below Average',
             5: 'Below Average', 6: 'Below Average', 7: 'Below Average', 8: 'Below Average', 9: 'Below Average',
             10: 'Average', 11: 'Above Average', 12: 'Above Average', 13: 'Above Average', 14: 'Above Average',
             15: 'Above Average', 16: 'Above Average', 17: 'Above Average', 18: 'Above Average', 19: 'Above Average',
             20: 'Above Average',
             }

print()
# for cols in df.columns:
# if df[cols].dtypes == 'object':
# cat_cols = cols
cat_cols = df.select_dtypes(exclude=int)
print()
num_cols = df.select_dtypes(include=int)

# explore the numerical side of things
df_corr = num_cols.corr()
print(df_corr['G3'].sort_values(ascending=False))

# for i in df_corr.columns:
# plt.figure(figsize=(8, 8))
# sns.regplot(x=df_corr['G3'], y=df_corr[i])
# plt.show()

for cols in num_cols.columns:
    print(num_cols[cols].value_counts())
print()

print(df['G3'].value_counts())
print()
print(df['G2'].value_counts())

data = df.copy()
data['G1'] = data['G1'].map(score_map)
data['G2'] = data['G2'].map(score_map)
data['G3'] = data['G3'].map(score_map)


print(data['G2'].unique())
print()
print(data['G1'].unique())
print()
print(data['G3'].unique())
print(data.isnull().sum().sort_values(ascending=False))

# for cols in cat_cols:
# plt.figure(figsize=(8, 8))
# sns.countplot(cat_cols[cols], hue=data['G3'], palette=sns.color_palette("cubehelix"))
# plt.show()

# Categorical Features
print(cat_cols.columns)
cat_list = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

lb = LabelEncoder()
data['G1'] = lb.fit_transform(data['G1'])
data['G2'] = lb.fit_transform(data['G2'])
data['G3'] = lb.fit_transform(data['G3'])

print(data['G1'].value_counts())
print()

for cols in cat_cols:
    cat_cols[cols] = lb.fit_transform(cat_cols[cols])
print(cat_cols.info())
# merging the two data frames together

full_df = pd.concat([cat_cols, data], axis=1)
print(type(full_df))
print()

# separate the target from the data

label = full_df['G3']
df_ = full_df.drop('G3', axis='columns')

print(label.head())
print(df_.info())

x_train, x_test, y_train, y_test = train_test_split(df_, label, test_size=0.3, random_state=41)

print('the length of the train data is', len(x_train))
print()
print('the length of the test data is', len(x_test))

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print()
print('the accuracy of logistic regression is set at')
print(log_reg.score(x_train, y_train))
prediction = log_reg.predict(x_test)
print()
print('the accuracy score of the test data for logistic regression is set at')
print(accuracy_score(y_test, prediction))

print()
print('Next up is the Support Vector Machines')
svc = SVC()
svc.fit(x_train, y_train)
print('the train accuracy for SVC is set at')
print(svc.score(x_train, y_train))
svc_prediction = svc.predict(x_test)
print()
print('the accuracy for the test is set at')
print(accuracy_score(y_test, svc_prediction))

print()
print('Random Forest')
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
print()
print('The train accuracy is set at')
print(rfc.score(x_train, y_train))
print('the test accuracy is ste at')
rfc_prediction = rfc.predict(x_test)
print(accuracy_score(y_test, rfc_prediction))

print()
print('Decision Trees')
dsc = DecisionTreeClassifier()
dsc.fit(x_train, y_train)
print('the train accuracy of decision tree is set at')
print(dsc.score(x_train, y_train))
print()
print('the test accuracy is set at')
dsc_prediction = dsc.predict(x_test)
print(accuracy_score(y_test, dsc_prediction))
