import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sales_file = pd.read_csv('sales.csv')

month = sales_file.iloc[:, 0:1]
sale_amount = sales_file.iloc[:, 1:2]

x_train, x_test, y_train, y_test = train_test_split(month, sale_amount, test_size=0.33, random_state=0)
standart_scaler = StandardScaler()


linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

prediction = linear_reg.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, prediction)
plt.show()