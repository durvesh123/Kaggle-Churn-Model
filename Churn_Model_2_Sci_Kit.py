import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


#loading train data set

data = pd.read_csv('/Users/durveshvedak/Downloads/Kaggle Churn/train.csv',header=0)

data.COLLEGE = data.COLLEGE.replace({"one":int(1),"zero":int(0)})
data.REPORTED_SATISFACTION = data.REPORTED_SATISFACTION.replace(
                                    {"very_unsat":int(5),
                                    "unsat":int(4),
                                    "avg":int(3),
                                    "sat":int(2),
                                    "very_sat":int(1)}
                                    )
data.REPORTED_USAGE_LEVEL = data.REPORTED_USAGE_LEVEL.replace(
                                    {"very_little":int(3),
                                    "little":int(5),
                                    "avg":int(1),
                                    "high":int(2),
                                    "very_high":int(4)}
                                    )
data.CONSIDERING_CHANGE_OF_PLAN = data.CONSIDERING_CHANGE_OF_PLAN.replace(
                                    {"actively_looking_into_it":int(4),
                                    "considering":int(5),
                                    "perhaps":int(1),
                                    "never_thought":int(2),
                                    "no":int(3)}
                                    )

print(data.head(10))                                    
