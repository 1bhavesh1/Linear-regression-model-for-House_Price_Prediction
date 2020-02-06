import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read csv 
dataset = pd.read_csv("C:\\Users\\bhavesh\\Documents\\datasets\\houseprices.csv")
size = dataset['Living Area']
price = dataset['Price']

# converting into arrays
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

#training
model = LinearRegression()
model.fit(x,y)

#MSE and R value
regression_model_mse = mean_squared_error(x,y)
print('MSE : ', math.sqrt(regression_model_mse))
print('R squared value : ',model.score(x,y))

# b0 and b1 parameters
print('b0 i.e intercept : ',model.coef_[0])
print('b1 i.e slope : ',model.intercept_[0])

#visualize the dataset
plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x),color='black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

#predicting new house price
print('Predicting new house price whose area is 2000 : ',model.predict([[2000]]))