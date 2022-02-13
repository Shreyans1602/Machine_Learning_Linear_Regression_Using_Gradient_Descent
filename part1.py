#Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as matplt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Load Data Set
# dataframe = pd.DataFrame(pd.read_excel("https://github.com/Shreyans1602/Machine_Learning_Linear_Regression/raw/main/Real_Estate_Valuation_Data_Set.xlsx", sheet_name= 'Sheet1',index_col='No'))
dataframe = pd.DataFrame(pd.read_excel("./Machine_Learning_Linear_Regression/Real_Estate_Valuation_Data_Set.xlsx", sheet_name= 'Sheet1',index_col='No'))

#Loaded Successfully
print("\nData Loaded Successfully")
print("Real Estate Valuation Data Set has {} data points with {} variables each.\n".format(*dataframe.shape))

#Pre-Processing Stage
#Check for null values in the dataframe
print("Pre-Processing the Data:")
print("Null entries found?: ",("No" if dataframe.isnull().sum().sum() == 0 else "Yes"))
#Check for duplicate values in the dataframe
print("Duplicate entries found?: ",("No\n" if dataframe.duplicated().sum() == 0 else "Yes\n"))
print("Check for categorical values:")
print(dataframe.dtypes)

#Rename attributes and describe the dataframe
print("\nRenaming the attributes for convenience. The dataframe is as follows:\n")
dataframe.rename(
    columns={
        "X1 transaction date": "Transaction_Date", 
        "X2 house age": "House_Age", 
        "X3 distance to the nearest MRT station": "Distance",
        "X4 number of convenience stores": "Num_Stores_NearBy",
        "X5 latitude": "Latitude",
        "X6 longitude": "Longitude",
        "Y house price of unit area": "Target",
    },
    inplace = True
)

print(dataframe.head())
print("\nDescription of the dataframe is as follows:")
print(dataframe.describe())

#Printing correlation matrix
print("\nCorrelation matrix is as follows:")
print(dataframe.corr())
print("\nMost impactful attributes on target variable are shows below in decending order:")
print(abs(dataframe.corr())['Target'].sort_values(ascending=False))

###################################################################################################

X = np.array(dataframe['Distance'])#.reshape(-1,1)
Y = np.array(dataframe['Target'])#.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=99)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialise the Scaler 
#scaler = StandardScaler()
  
# To scale data 
#X_train_Scaled = scaler.fit_transform(X_train) 
# y_train_Scaled = scaler.fit_transform(y_train) 

#X_test_Scaled = scaler.transform(X_test) 
# y_test_Scaled = scaler.transform(y_test) 

#lr = LinearRegression()
#lr.fit(X_train, y_train)

#mse = mean_squared_error(y_test,lr.predict(X_test))
#print(mse)
#r2 = r2_score(y_test,lr.predict(X_test))
#print(r2)

# Rasesh's edit
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#print(X_train.shape)
#print(y_train.shape)
print(y_train.reshape(len(y_train)).shape)
clf.fit(X_train, y_train)
#clf.fit([[0., 0.], [1., 1.]], [0, 1])
# SGDClassifier(max_iter=5)
# mseSGD = mean_squared_error(y_test,clf.predict(X_test))
# print(mseSGD)

# Plot outputs
# matplt.figure(figsize=(10,7))
# matplt.scatter(x=dataframe['Distance'], y=dataframe['Target'])
# matplt.plot(X_test, lr.predict(X_test), color='red')
# matplt.xlabel('The distance to the nearest MRT station (unit: meter)')
# matplt.ylabel('House price of unit area')
# matplt.title('Real Estate Valuation - Linear Regression Prediction')
# matplt.show()

#Generate plots for information extraction
# heatmap = matplt.figure()
# heat_reference = heatmap.add_subplot(111)
# cax = heat_reference.matshow(dataframe.corr(), interpolation='nearest')
# heatmap.colorbar(cax)
# att_names = dataframe.columns.values

# heat_reference.set_xticklabels(['']+att_names)
# heat_reference.set_yticklabels(['']+att_names)

# matplt.show()

