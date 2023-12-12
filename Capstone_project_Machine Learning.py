#import libaries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.io import arff



# Load ARFF file using scipy
arff_file_path = '/Users/nazira/Downloads/dataset_37_diabetes.arff'
data, meta = arff.loadarff(arff_file_path)
df = pd.DataFrame(data)
print(df.head(20))



# convert string value to numeric value
df['class'] = df['class'].map({b'tested_positive': 1, b'tested_negative': 0})
print(df.head(5))



#define and reshape X_train and y_train 
X = np.array(df[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']])
y = np.array(df['class'])

X = X.reshape((768, 8))  

print(f"the shape of the inputs X is: {X.shape}")
print(f"the shape of the targets y is: {y.shape}")



#split the dataset into training, CV , test sets
x_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

print(f"the shape of the inputs X is: {X.shape}")
print(f"the shape of the targets y is: {y.shape}")
print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross-validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross-validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

del x_, y_


#Feature scaling
scaler_linear = StandardScaler()
x_train_scaled = scaler_linear.fit_transform(x_train)



# Train model
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)


# evaluate the trained model
yhat = linear_model.predict(x_train_scaled)

print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")


# calculate MSE with cross validation data
x_cv_scaled = scaler_linear.transform(x_cv)

# Feed the scaled cross validation set
yhat = linear_model.predict(x_cv_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")



