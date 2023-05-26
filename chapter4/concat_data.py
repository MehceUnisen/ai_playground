import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def get_data():

    csv_file = pd.read_csv('data.csv')

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    age = imputer.fit_transform(csv_file.iloc[:,3:4])

    label_encoder = LabelEncoder()
    one_hot_enc = OneHotEncoder()

    country = csv_file.iloc[:,0:1].values
    country[:,0] = label_encoder.fit_transform(csv_file.iloc[:,0])
    country = one_hot_enc.fit_transform(country).toarray()

    gender = csv_file.iloc[:,4:5]
    gender[:,0] = label_encoder.fit_transform(csv_file.iloc[:,4:5])
    gender = one_hot_enc.fit_transform(gender).toarray()

    rest = csv_file.iloc[:,1:3].values

    df_country = pd.DataFrame(data=country, index=range(22), columns=['fr', 'tr', 'us'])
    df_age = pd.DataFrame(data=age, index=range(22),columns=['age'])
    df_gender = pd.DataFrame(data=gender.iloc[:, 0:1], index=range(22), columns=['gender'])
    df_rest = pd.DataFrame(data=rest, index=range(22), columns=['length', 'weight'])

    result_data = pd.concat([df_country, df_rest, df_age, df_gender], axis=1)
    return result_data