# importing necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.metrics import classification_report  
from sklearn.preprocessing import MinMaxScaler 
import joblib 



#first 5 rows to be printed, df.tail()
print("first 5 row: ")
print(df.head())

print(df.info())

print("print 5 columns")
print(df.columns)

df=df.drop('unnamed: 0', axis=1)
print(df.head())

print(df.describe()) # Statistics of the dataset



# -------------------------------
# STEP 2: DEFINE FEATURES AND LABELS
# -------------------------------
print("print columns 0 to 9 ")
X = df.iloc[:, 0:20]   # This gives you columns 0 to 19 (sensor_0 to sensor_19)

print("print 0 to 19 columns")
y = df.iloc[:, 20:]
print("sample x:")
print(X.sample(10))

print(y.sample(10))


print("information of x:")
print(X.info())
print("impormation of y:")
print(y.info())

print(X)

print(X.shape, y.shape)










