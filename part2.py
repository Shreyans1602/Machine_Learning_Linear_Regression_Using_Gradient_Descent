#Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Load Data Set
#dataset = pd.read_excel("https://github.com/Shreyans1602/Machine_Learning_Linear_Regression/raw/main/Real_Estate_Valuation_Data_Set.xlsx", sheet_name= 'Sheet1',index_col='No')
dataset = pd.read_excel("./Machine_Learning_Linear_Regression/Real_Estate_Valuation_Data_Set.xlsx", sheet_name= 'Sheet1',index_col='No')
dataframe = pd.DataFrame(dataset)
#Loaded Successfully
print("\nData Loaded Successfully")
print("Real Estate Valuation Data Set has {} data points with {} variables each.\n".format(*dataset.shape))

#Pre-Processing Stage
#Check for null values in the dataset
print("Pre-Processing the Data:")
print("Null entries found?: ",("No" if dataset.isnull().sum().sum() == 0 else "Yes"))
#Check for duplicate values in the dataset
print("Duplicate entries found?: ",("No\n" if dataset.duplicated().sum() == 0 else "Yes\n"))
print("Check for categorical values:")
print(dataset.dtypes)

#Rename attributes and describe the dataset
print("\nRenaming the attributes for convinience. The dataset is as follows:\n")
dataset.rename(
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
print(dataset.head())
print("\nDescription of the Dataset is as follows:")
print(dataset.describe())

#Printing correlation matrix
print("\nCorrelation matrix is as follows:")
print(dataset.corr())
print("\nMost impactful attributes on target variable are shows below in decending order:")
print(abs(dataset.corr())['Target'].sort_values(ascending=False))

###################################################################################################

X = np.array(dataframe['Distance']).reshape(-1,1)
Y = np.array(dataframe['Target']).reshape(-1,1)

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=99)

y_train = y_train.reshape(len(y_train),)
y_valid = y_valid.reshape(len(y_valid),)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# Initialise the Scaler 
scaler = StandardScaler() 
  
# To scale data 
X_train_Scaled = scaler.fit_transform(X_train) 
# y_train_Scaled = scaler.fit_transform(y_train) 

X_valid_Scaled = scaler.transform(X_valid) 
# y_valid_Scaled = scaler.transform(y_valid) 

lr = LinearRegression()
lr.fit(X_train, y_train)

mse = mean_squared_error(y_valid,lr.predict(X_valid))
print(mse)
r2 = r2_score(y_valid,lr.predict(X_valid))
print(r2)

# Plot outputs
plt.figure(figsize=(10,7))
plt.scatter(x=dataframe['Distance'], y=dataframe['Target'])
plt.plot(X_valid, lr.predict(X_valid), color='red')
plt.xlabel('The distance to the nearest MRT station (unit: meter)')
plt.ylabel('House price of unit area')
plt.title('Real Estate Valuation - Linear Regression Prediction')

plt.show()



#Generate plots for information extraction
# heatmap = plt.figure()
# heat_reference = heatmap.add_subplot(111)
# cax = heat_reference.matshow(dataset.corr(), interpolation='nearest')
# heatmap.colorbar(cax)
# att_names = dataframe.columns.values

# heat_reference.set_xticklabels(['']+att_names)
# heat_reference.set_yticklabels(['']+att_names)

# plt.show()

