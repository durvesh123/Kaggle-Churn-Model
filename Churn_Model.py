import numpy as np
import pandas as pd


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



#loading trainig data from data
train_data_X = data.iloc[0:15000,0:11] #15000*11
train_data_Y = data.iloc[0:15000,11]
train_data_Y = train_data_Y.values.reshape(train_data_Y.shape[0],1) #15000*1



#load test data set from data
test_data_X = data.iloc[15000:18001,0:11]#3000*11
test_data_Y = data.iloc[15000:18001,11]
test_data_Y = test_data_Y.values.reshape(test_data_Y.shape[0],1) #3000*1




num_training_examples = train_data_X.shape[0] #15000
num_test_examples = test_data_X.shape[0] #3000

#Sigmoid function
def sigmoid(z):
    s = 1/(1+np.exp(z*(-1)))
    return s



#Initialize paramter w and b
def initialize_with_zeros(dim): #dim is the number of parameters
    w = np.zeros((dim,1))
    b = 0
    return w, b

#Implements forward and backward propagation
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w-learning_rate*dw
        b = b-learning_rate*db

        if i % 1000 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs



def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_data_X, train_data_Y, test_data_X, test_data_Y, num_iterations = 1, learning_rate = 0.005, print_cost = True)


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')
