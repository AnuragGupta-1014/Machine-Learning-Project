                 # CUSTOMER SEGMENTATION
import os
import pandas as pd 
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression                
 

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

                                   # Data Collection

data_path = 'Customer_Segmentation.csv'
if os.path.exists(data_path):
    Data = pd.read_csv(data_path)
    print(Data.head())
    print(Data.tail())
    print("Shape of Datasheet:", Data.shape)
    print("Columns of Datasheet:", Data.columns)
    print(Data.index)
    print("\n")
    print(Data.loc[[1, 2]])
else:
    print(f"File {data_path} not found.")

                                  # Data Preprocessing  
print(Data.isna()) 
print(Data.isna().sum())
print(Data.head().dropna())
print("\n")
print(Data.tail().sort_values)
                  
                                  # Data Analysis
# Scatter Plot
ploting = Data.head(100).plot(kind="scatter",x= "tenure", y= "MonthlyCharges") 
plt.title("Customer Segmentation")                             
plt.xlabel("Tenure")                              
plt.ylabel("MonthlyCharges")
plt.show()

# Pie plot
df = Data.head(20)["tenure"].plot(kind= "pie")
plt.show()  

# Line plot
Data['MonthlyCharges'] = pd.to_numeric(Data['MonthlyCharges'], errors='coerce')
Data['TotalCharges'] = pd.to_numeric(Data['TotalCharges'], errors='coerce')
Data.head(40).plot(kind="line", x="tenure", y="TotalCharges")
plt.xlabel("tenure")
plt.ylabel("Total Charges")
plt.title("Tenure vs Total Charges")
plt.show()

                         # Exploratory Data Analysis (EDA) - (Missing Values And Outlier)

print(Data["MonthlyCharges"].isnull().sum())                             # Before Filling
avg = Data['MonthlyCharges'].mean()
print("Average of MonthlyCharges columns = ", avg,"\n")                  # filling missing value with avg   
Data['MonthlyCharges'] = Data['MonthlyCharges'].fillna(avg)
print(Data["MonthlyCharges"].isnull().sum())                             # After Filling
           
           
           
# Outlier
sns.boxplot(y = Data["tenure"])
plt.show()
sns.boxenplot(x =Data["TotalCharges"])
plt.show()

                                # Feature Engineering
 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler_standard = StandardScaler()
Data_standardized = scaler_standard.fit_transform(Data[['tenure', 'MonthlyCharges', 'TotalCharges']])
Data_standardized = pd.DataFrame(Data_standardized, columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

# Normalization
scaler_minmax = MinMaxScaler()
Data_normalized = scaler_minmax.fit_transform(Data[['tenure', 'MonthlyCharges', 'TotalCharges']])
Data_normalized = pd.DataFrame(Data_normalized, columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

# Plotting the standardized and normalized data

plt.subplot(2, 3, 1)
sns.histplot(Data_standardized['tenure'])
plt.title('Standardized Tenure')

plt.subplot(2, 3, 2)
sns.histplot(Data_standardized['MonthlyCharges'])
plt.title('Standardized MonthlyCharges')

plt.subplot(2, 3, 3)
sns.histplot(Data_standardized['TotalCharges'])
plt.title('Standardized TotalCharges')


plt.tight_layout()
plt.show()

                                                # Model Building


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


Data['Churn'] = Data['Churn'].map({'Yes': 1, 'No': 0})


# Selecting features and target variable
X = Data_standardized[['TotalCharges', 'MonthlyCharges']]
y = Data['Churn']


# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a Random Forest model using standardized features
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)



# Evaluating the models
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print("Accuracy of Random Forest model using standardized  {0:0.4f}".format(accuracy))

