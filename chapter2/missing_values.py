import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('missing_data.csv')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age = data.iloc[:,1:4].values
# print(age)
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)