import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import explained_variance_score



#loading train data set

data = pd.read_csv('/Users/durveshvedak/Downloads/Kaggle Churn/train.csv',header=0)


data.COLLEGE = data.COLLEGE.replace({"one":int(1),"zero":int(0)})
data.REPORTED_SATISFACTION = data.REPORTED_SATISFACTION.replace(
                                    {"very_unsat":int(500),
                                    "unsat":int(400),
                                    "avg":int(300),
                                    "sat":int(200),
                                    "very_sat":int(100)}
                                    )
data.REPORTED_USAGE_LEVEL = data.REPORTED_USAGE_LEVEL.replace(
                                    {"very_little":int(500),
                                    "little":int(400),
                                    "avg":int(300),
                                    "high":int(200),
                                    "very_high":int(100)}
                                    )
data.CONSIDERING_CHANGE_OF_PLAN = data.CONSIDERING_CHANGE_OF_PLAN.replace(
                                    {"actively_looking_into_it":int(500),
                                    "considering":int(400),
                                    "perhaps":int(300),
                                    "never_thought":int(200),
                                    "no":int(100)}

                                    )


#load test data
data_test_all = pd.read_csv('/Users/durveshvedak/Downloads/Kaggle Churn/test.csv',header=0)

data_test_all.COLLEGE = data_test_all.COLLEGE.replace({"one":int(1),"zero":int(0)})
data_test_all.REPORTED_SATISFACTION = data_test_all.REPORTED_SATISFACTION.replace(
                                    {"very_unsat":int(500),
                                    "unsat":int(400),
                                    "avg":int(300),
                                    "sat":int(200),
                                    "very_sat":int(100)}
                                    )
data_test_all.REPORTED_USAGE_LEVEL = data_test_all.REPORTED_USAGE_LEVEL.replace(
                                    {"very_little":int(500),
                                    "little":int(400),
                                    "avg":int(300),
                                    "high":int(200),
                                    "very_high":int(100)}
                                    )
data_test_all.CONSIDERING_CHANGE_OF_PLAN = data_test_all.CONSIDERING_CHANGE_OF_PLAN.replace(
                                    {"actively_looking_into_it":int(500),
                                    "considering":int(400),
                                    "perhaps":int(300),
                                    "never_thought":int(200),
                                    "no":int(100)}
                                    )



# Training data
X = data.iloc[:,0:11]
Y = data.iloc[:,11]



#Normalize and Scale training set data
X = preprocessing.normalize(X)
X = preprocessing.scale(X)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)



#actual test
X_test = data_test_all.iloc[:,0:11]
X_test = preprocessing.normalize(X_test)
X_test = preprocessing.scale(X_test)


#Initialize Model
model = SVC(kernel='rbf',class_weight='balanced', C=2.0)
#weights = [2.0/3]*x_train.shape[0]
model.fit(x_train, y_train)


predictions = model.predict(x_test)

"""
predictions = model.predict(X_test)

print(predictions[:10])
print(len(predictions))
with open('/Users/durveshvedak/Downloads/Kaggle Churn/results.txt','w') as f:
    for i in predictions:
        f.write(str(i)+"\n")
"""

print(accuracy_score(y_test,predictions))
