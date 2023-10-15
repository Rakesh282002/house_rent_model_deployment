import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
df_hr=pd.read_csv(r"E:\AIML&DS INTERN\House_Rent_Dataset.csv")
df_hr_cpy=df_hr.copy()
df_hr_cpy1=df_hr.copy()
df_hr.head()
del df_hr["Posted On"]
del df_hr["Floor"]
del df_hr["Bathroom"]
del df_hr["Area Locality"]
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_hr["Area Type"]=le.fit_transform(df_hr["Area Type"])
#df_hr["Area Locality"]=le.fit_transform(df_hr["Area Locality"])

df_hr["City"]=le.fit_transform(df_hr["City"])
df_hr["Furnishing Status"]=le.fit_transform(df_hr["Furnishing Status"])
df_hr["Tenant Preferred"]=le.fit_transform(df_hr["Tenant Preferred"])
df_hr["Point of Contact"]=le.fit_transform(df_hr["Point of Contact"])
Q1 = df_hr.Size.quantile(0.05)
Q3 =df_hr.Size.quantile(0.95)
IQR = Q3 - Q1
df_hr = df_hr[(df_hr.Size >= Q1 - 1.5*IQR) & (df_hr.Size <= Q3 + 1.5*IQR)]
IndepVar = []
for col in df_hr.columns:
    if col != 'Rent':
        IndepVar.append(col)

TargetVar = 'Rent'

x = df_hr[IndepVar]
y = df_hr[TargetVar]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape 
from sklearn.ensemble import ExtraTreesRegressor
modelETR = ExtraTreesRegressor()
modelETR.fit(x_train, y_train)
import pickle
pickle.dump(modelETR,open("E:\ML Projects\hrp_model.pkl",'wb'))



