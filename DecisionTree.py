import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#loading train data set

data = pd.read_csv('/Users/durveshvedak/Downloads/Kaggle Churn/train.csv',header=0)


#Encoding Text values to numeric
le = preprocessing.LabelEncoder()
data.COLLEGE = le.fit(data.COLLEGE).transform(data.COLLEGE)
data.REPORTED_SATISFACTION = le.fit(data.REPORTED_SATISFACTION).transform(data.REPORTED_SATISFACTION)
data.REPORTED_USAGE_LEVEL = le.fit(data.REPORTED_USAGE_LEVEL).transform(data.REPORTED_USAGE_LEVEL)
data.CONSIDERING_CHANGE_OF_PLAN = le.fit(data.CONSIDERING_CHANGE_OF_PLAN).transform(data.CONSIDERING_CHANGE_OF_PLAN)



#load test data and encoding
data_test_all = pd.read_csv('/Users/durveshvedak/Downloads/Kaggle Churn/test.csv',header=0)
le = preprocessing.LabelEncoder()
data_test_all.COLLEGE = le.fit(data_test_all.COLLEGE).transform(data_test_all.COLLEGE)
data_test_all.REPORTED_SATISFACTION = le.fit(data_test_all.REPORTED_SATISFACTION).transform(data_test_all.REPORTED_SATISFACTION)
data_test_all.REPORTED_USAGE_LEVEL = le.fit(data_test_all.REPORTED_USAGE_LEVEL).transform(data_test_all.REPORTED_USAGE_LEVEL)
data_test_all.CONSIDERING_CHANGE_OF_PLAN = le.fit(data_test_all.CONSIDERING_CHANGE_OF_PLAN).transform(data_test_all.CONSIDERING_CHANGE_OF_PLAN)

#Standard Scaler
scaler = StandardScaler()

#actual test
X_test = data_test_all.iloc[:,0:11]
X_test = scaler.fit_transform(X_test)


#Train and Test Data
X = data.iloc[:,0:11]
Y = data.iloc[:,11]
X = scaler.fit_transform(X)



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


#Decision Tree

clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 42,
                               max_depth=4, min_samples_leaf=20,min_samples_split=2,class_weight='balanced')
clf_gini.fit(x_train, y_train)


y_pred = clf_gini.predict(x_test)

'''
with open('/Users/durveshvedak/Downloads/Kaggle Churn/results.txt','w') as f:
    for i in y_pred:
        f.write(str(i)+"\n")
'''
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)







