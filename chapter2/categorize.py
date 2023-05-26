import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('data.csv')
country = data.iloc[:,0:1].values
label_encoding = preprocessing.LabelEncoder()
country[:,0] = label_encoding.fit_transform(data.iloc[:,0:1])


one_hot_enc = preprocessing.OneHotEncoder()
country = one_hot_enc.fit_transform(country).toarray()

print(country)