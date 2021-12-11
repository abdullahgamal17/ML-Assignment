#libraries
import numpy as np
from numpy.core.fromnumeric import shape 
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Functions
def Normalize(X):
    
    """[summary]
    takes a matrix X and normalizes it in this way : (X - min(X)/(range X))
    
    Args:
        X ([2d array]): [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows]
        
    Returns:
     normalized_X [2d array]: [Normalized X matrix of shape same as X (no_of_features,no_of_rows)]
    """
    minn = X.min(axis=1)
    maxx = X.max(axis=1)
    normalized_X = copy.deepcopy(X)
    nx = X.shape[0]
    for i in range(nx):
        normalized_X[i] = (normalized_X[i]-minn[i])/(maxx[i]-minn[i])
    
    return normalized_X

def initialize_zeros(nx):
    """[summary]
    Initialises W (Weights matrix) with zeros having shape of (nx,1) "column vector" and
    b (bias) "W0X0" initialised with zero "scalar" 
    Args:
        nx ([int]): [no_of_features]

    Returns:
        Weights [2d array]: [Weights matrix of shape (nx,1) of zeros "float64"]
        [type]: [scalar "float64"]
    """
    Weights = np.zeros(shape=(nx,1) , dtype= float)
    b = 0.0
    return Weights , b

def sigmoid(z):
    """[summary]
    Applies sigmod function to the input z

    Args:
        z ([2d array]): [arrise after activation "z = np.dot(Weights.T , X) + b"]

    Returns:
        s[2d array]: [z after sigmoid function "s = 1/(1+np.exp(-z))"]
    """
    x = np.exp(-z)
    s = 1/(1+x)
    
    return s

def activate(X,Weights,b):
    """[summary]
    Creates z matrix (z = np.dot(Weights.T , X) + b) and applies sigmoid function to it

    Args:
        X ([2d array]): [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows]
        Weights ([2d array]): [matrix of shape (nx,1) where nx is no_of_features]
        b (float): [scalar "float64"]

    Returns:
        A[2d array]: [matrix of shape (1,m) where m is no_of_rows , arises after applying sigmoid function]
    """
    z = np.dot(Weights.T , X) + b
    A = sigmoid(z)
    return A
    
def predict(X,Weights,b):
    """[summary]
    predicts the class of the input X by applying sigmoid function to the input z then rounding it to 0 or 1 according to a threshold of 0.5
    Args:
        X ([2d array]): [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows]
        Weights ([2d array]): [matrix of shape (nx,1) where nx is no_of_features]
        b ([float]): [scalar "float64"]

    Returns:
        y_hat[2d array]: [matrix of shape (1,m) where m is no_of_rows , made of 0s and 1s "True or False"]
    """
    A = activate(X,Weights,b)
    y_hat = A > 0.5    
    return y_hat

def cost_function(A,Y):
    """[summary]
        Calculates the cost of the model , using the formula : cost = -1/m * np.sum( np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), (1-Y.T))) 
    Args:
        A[2d array]: [matrix of shape (1,m) where m is no_of_rows , arises after applying sigmoid function]
        Y[2d array]: [matrix of shape (1,m) where m is no_of_rows , is the true value "0 or 1"]

    Returns:
        cost [float]: [represents the cost of the model]
    """
    m = Y.shape[1]
    cost = -1/m * np.sum( np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), (1-Y.T))) #from andrew ng course "Deep learning and Neural networks (Coursera)"
    return cost
    
def gradient_descent(X,Weights,b,Y,learning_rate,iterations,print_cost=False):
    """[summary]
    performs gradient descent on the model

    Args:
        X ([2d array]): [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows]
        Weights ([2d array]): [matrix of shape (nx,1) where nx is no_of_features]
        b ([float]): [scalar "float64"]
        Y[2d array]: [matrix of shape (1,m) where m is no_of_rows , is the true value "0 or 1"]
        learning_rate ([float]): [learning rate of the model in which the variables are updated]
        iterations ([int]): [number of iterations performing gradient descent]
        print_cost ([bool]): [whether to print the cost or not]

    Returns:
        Weights ([2d array]): [matrix of shape (nx,1) where nx is no_of_features]
        b (float): [scalar "float64"]
        cost_history[array]: [contains the history of costs during gradient descent]
    """
    m = X.shape[1]
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        A = activate(X,Weights,b)
        cost = cost_function(A,Y)
        cost_history[i] = cost
        
        ###############################
        dz = A - Y
        dw = (1/m)*(np.dot(X,dz.T))
        db = (1/m)*np.sum(dz)
        ###############################
        # This method was explained in Andrew Ng's course on coursera regarding logistic regression" 
        Weights -= learning_rate*dw
        b -= learning_rate*db
        if(print_cost):
            if(i%100 == 0):
                print(f"The cost is {cost} ")

    return Weights,b,cost_history

def count_errors(y_hat,Y):
    """[summary]

    Args:
        y_hat[2d array]: [matrix of shape (1,m) where m is no_of_rows , made of 0s and 1s "True or False" "Predicted Values"]
        Y[2d array]: [matrix of shape (1,m) where m is no_of_rows , is the true value "0 or 1" "Real Value"]

    Returns:
        errors[int]: [number of wrong predictions]
    """
    errors = np.sum(np.abs(y_hat - Y))
    return errors

def split(X,Y,test_size):
    
    """[summary]
    This splits our data into training and test dataset
    Args:
        X ([2d array]): [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows]
        Y[2d array]: [matrix of shape (1,m) where m is no_of_rows , is the true value "0 or 1" "Real Value"]
        test_size[float]: [between 0 and 1 , represents the percentage of the data used for testing]
    
    
    Returns:
        X_train[2d array]: [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows dedicated to training]
        X_test[2d array]: [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows dedicated to testing]
        Y_train[2d array]: [matrix of shape (1,m) where m is no_of_rows dedicated to training , is the true value "0 or 1" "Real Value"]
        Y_test[2d array]: [matrix of shape (1,m) where m is no_of_rows dedicated to testing , is the true value "0 or 1" "Real Value"]
    """
    number_of_elements = X.shape[1]
    number_of_rows = X.shape[0]
    all_indices = np.arange(number_of_elements)
    trainindices = np.random.choice(all_indices,int(number_of_elements*(1-test_size)),replace=False)
    testindices = np.setdiff1d(all_indices,trainindices)
    X_train = X[:,trainindices]
    X_test = X[:,testindices]
    Y_train = Y[:,trainindices]
    Y_test = Y[:,testindices]
    return X_train,X_test,Y_train,Y_test
#logistic model
def logistic_model(X,Y,learning_rate,iterations,print_cost=False):
    """[summary]
        This is the main function of the logistic regression model
    Args:
        X ([2d array]): [matrix of shape (nx,m) where nx is no_of_features and m is no_of_rows]
        Y[2d array]: [matrix of shape (1,m) where m is no_of_rows , is the true value "0 or 1"]
        learning_rate ([float]): [learning rate of the model in which the variables are updated]
        iterations ([int]): [number of iterations performing gradient descent]
        print_cost ([bool]): [whether to print the cost or not]

    Returns:
        Weights ([2d array]): [matrix of shape (nx,1) where nx is no_of_features]
        b (float): [scalar "float64"]
        cost_history[array]: [contains the history of costs during gradient descent]
    """
    nx = X.shape[0]
    m = X.shape[1]
    Weights , b = initialize_zeros(nx)
    Weights,b,cost_history = gradient_descent(X,Weights,b,Y,learning_rate,iterations,print_cost)
    return Weights,b,cost_history



# Code starts here ( like main in C++)
#help for myself
#shape of X = (nx,m) = (4,303)
#shape of weights = (nx,1) , W.T shape = (1,nx) row vector
#shape of Y = (1,m) = (1,303)
#b is a number

# dataframes of interest
df = pd.read_csv("heart.csv")
columns = df.columns
columns_of_interest = ["trestbps","chol","thalach","oldpeak"]
df_interested = df[columns_of_interest]
df_output = df["target"]

#matrices needed (X,X_normalized,Y) and values needed (m,learning_rate,iterations)
X = df_interested.values.T
X_normalized = Normalize(X)
m = X.shape[1]
Y = df_output.values.reshape(1,m)
learning_rate = 0.3
iterations = 1500

#split into training and test sets
X_train,X_test,Y_train,Y_test = split(X,Y,0.2)

#Normalizing X_train , X_test
X_train_normalized = Normalize(X_train)
X_test_normalized = Normalize(X_test)

#sklearn model
model = LogisticRegression()
X_train_sklearn , X_test_sklearn , Y_train_sklearn , Y_test_sklearn = train_test_split(df_interested,df_output,test_size = 0.2)
#training on X_train_sklearn , Y_train_sklearn
model.fit(X_train_sklearn,Y_train_sklearn)
#accuracy on X_test_sklearn , Y_test_sklearn
sk_accuaracy = model.score(X_test_sklearn,Y_test_sklearn)*100
print(f"The sklearn model accuaracy is {sk_accuaracy} %")

print(X_train_normalized)
#my_model
#training my model on X_train , Y_train
Weights,b,cost_history = logistic_model(X_train_normalized,Y_train,learning_rate,iterations,print_cost=True)
#accuracy on X_test , Y_test
my_model_predictions = predict(X_test_normalized,Weights,b)
errors = count_errors(my_model_predictions,Y_test)
accuaracy = (1-(errors/m))*100
print(f"The accuracy of my model is {accuaracy} %")
#plotting cost history of my_model
plt.plot(cost_history)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("Cost history of my model")
plt.show()



