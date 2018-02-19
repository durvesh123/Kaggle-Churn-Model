import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





#loading train data set

data = pd.read_csv('/Users/durveshvedak/Downloads/Kaggle Churn/train.csv',header=0)


#Encoding Text values to numeric
le = preprocessing.LabelEncoder()
data.COLLEGE = le.fit(data.COLLEGE).transform(data.COLLEGE)
data.REPORTED_SATISFACTION = le.fit(data.REPORTED_SATISFACTION).transform(data.REPORTED_SATISFACTION)
data.REPORTED_USAGE_LEVEL = le.fit(data.REPORTED_USAGE_LEVEL).transform(data.REPORTED_USAGE_LEVEL)
data.CONSIDERING_CHANGE_OF_PLAN = le.fit(data.CONSIDERING_CHANGE_OF_PLAN).transform(data.CONSIDERING_CHANGE_OF_PLAN)


X = data.iloc[:,0:11]
Y = data.iloc[:,11]


#Normalize and Scale training set data
X = preprocessing.normalize(X)
X = preprocessing.scale(X)


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#Initialize Model
model = SVC(kernel='rbf',class_weight='balanced', C=2.0)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(accuracy_score(predictions,y_test))
