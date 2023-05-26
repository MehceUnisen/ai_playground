import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import concat_data

data = concat_data.get_data()

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:6], data.iloc[:, 6:], test_size=0.33, random_state=0)

standart_scaler = StandardScaler()
scaled_x_train = standart_scaler.fit_transform(x_train) 
scaled_x_test = standart_scaler.fit_transform(x_test)

