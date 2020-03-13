import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
from keras import utils
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

maths = df_math.drop(index=0, axis=0)
print()
print('the data for the people offering maths')
print(maths.head())
df = maths.copy()

int_features = ['age', 'Fedu', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1',
                'G2', 'G3', 'freetime', 'goout', 'health', 'Medu', 'studytime', 'traveltime']

lb = LabelEncoder()

for vals in int_features:
    df[vals] = df[vals].astype(int)

score_map = {0: 'Below Average', 1: 'Below Average', 2: 'Below Average', 3: 'Below Average', 4: 'Below Average',
             5: 'Below Average', 6: 'Below Average', 7: 'Below Average', 8: 'Below Average', 9: 'Below Average',
             10: 'Average', 11: 'Above Average', 12: 'Above Average', 13: 'Above Average', 14: 'Above Average',
             15: 'Above Average', 16: 'Above Average', 17: 'Above Average', 18: 'Above Average', 19: 'Above Average',
             20: 'Above Average',
             }
df['G3'] = df['G3'].map(score_map)
df['G2'] = df['G2'].map(score_map)
df['G1'] = df['G1'].map(score_map)

cat_list = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for cats in cat_list:
    df[cats] = lb.fit_transform(df[cats])

df['G1'] = lb.fit_transform(df['G1'])
df['G2'] = lb.fit_transform(df['G2'])
df['G3'] = lb.fit_transform(df['G3'])

print(df.info())

# separate the target and the data
labels = df['G3']
data = df.drop('G3', axis='columns')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

print('the length of the train data is', len(x_train))
print()
print('the length of the test data is', len(x_test))

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
input_dim = len(x_train.columns)
print()
print('the accuracy of logistic regression is set at')
print(log_reg.score(x_train, y_train))
prediction = log_reg.predict(x_test)
print()
print('the accuracy score of the test data for logistic regression is set at')
print(accuracy_score(y_test, prediction))
print()
print('the confusion matrix is set at')
print(confusion_matrix(y_test, prediction))
print()

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
rfc = RandomForestClassifier(max_depth=4, max_features=8)
rfc.fit(x_train, y_train)
print()
print('The train accuracy is set at')
print(rfc.score(x_train, y_train))
print('the test accuracy is ste at')
rfc_prediction = rfc.predict(x_test)
print(accuracy_score(y_test, rfc_prediction))

print()
print('Decision Trees')
dsc = DecisionTreeClassifier(max_depth=4, max_features=8)
dsc.fit(x_train, y_train)
print('the train accuracy of decision tree is set at')
print(dsc.score(x_train, y_train))
print()
print('the test accuracy is set at')
dsc_prediction = dsc.predict(x_test)
print(accuracy_score(y_test, dsc_prediction))


y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=input_dim, kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(150, activation='relu', kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(200, activation='relu', kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(150, activation='relu', kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(100, activation='relu', kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(25, activation='relu', kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(10, activation='relu', kernel_initializer='random_uniform'))
# model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=2, epochs=200, validation_data=(x_test, y_test))
# callbacks=[EarlyStopping(monitor='val_acc', patience=20)])
print()
# deep_preidction = model.predict(x_test)
print()
# print('the accuracy of the deep learning model is set at: ')
# print(accuracy_score(y_test, deep_preidction))

# second trial at building this model for better accuracy
net = Sequential()
net.add(Dense(10, activation='relu', input_dim=input_dim, kernel_initializer='random_uniform'))
net.add(Dense(30, activation='relu', kernel_initializer='random_uniform'))
net.add(Dense(50, activation='relu', kernel_initializer='random_uniform'))
net.add(Dense(60, activation='relu', kernel_initializer='random_uniform'))
net.add(Dense(80, kernel_initializer='random_uniform', activation='relu'))
net.add(Dense(30, kernel_initializer='random_uniform', activation='relu'))
net.add(Dense(12, activation='relu', kernel_initializer='random_uniform'))
net.add(Dense(3, activation='softmax'))

net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
net.fit(x_train, y_train, batch_size=2, epochs=30, validation_data=(x_test, y_test))
