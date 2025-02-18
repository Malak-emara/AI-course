import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

lb=LabelEncoder()
imputer=SimpleImputer(missing_values=np.nan,strategy='mean') 

df= pd.read_csv('healthcare-dataset-stroke-data.csv')
df.drop(['id'],axis=1,inplace=True)

df['bmi']=imputer.fit_transform(df[['bmi']])

for col in ['work_type','Residence_type','smoking_status']:
    print(f'number of column {col} is : ',df[col].nunique())
    print(f'name of column {col} is : ',df[col].unique())

df['work_type'] = lb.fit_transform(df['work_type'])
df['Residence_type'] = lb.fit_transform(df['Residence_type'])
df['smoking_status'] = lb.fit_transform(df['smoking_status'])
df['gender'] = lb.fit_transform(df['gender'])
df['ever_married'] = lb.fit_transform(df['ever_married'])


my_scaler = MinMaxScaler(feature_range=(0,1))
df_scaler_minMax = my_scaler.fit_transform(df)
df = pd.DataFrame(df_scaler_minMax, columns=df.columns)  

print(df.head(10))