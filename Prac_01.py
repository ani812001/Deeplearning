#Understanding Tensor Slicing 
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
_____________________________________________________________________________________________
print(train_images.shape)
my_slice = train_images[10]
print(my_slice.shape) # select the tenth image from the dataset of 60K Image 
____________________________________________________________________________________________________
my_slice = train_images[10:100]
print(my_slice.shape) # select the tenth image from the dataset of 60K Image
__________________________________________________________________________________________

my_slice = train_images[10:100, 2:28 , 0:28]
print(my_slice.shape) # select the tenth image from the dataset of 60K Image
______________________________________________________________________________________________
my_slice = train_images[10:100, : , :]
print(my_slice.shape) # select the tenth image from the dataset of 60K Image
_________________________________________________________________________________________________

my_slice = train_images[:14:28, 14:28]
my_slice.shape
_____________________________________________________________________________________________________
my_slice = train_images[:,7:21,7:21]
my_slice.shape
_____________________________________________________________________________________________________

n = 0 
batch_n = train_images[n*128:(n+1)*128]
batch_n.shape
______________________________________________________________________________________________________________

n =1 
batch_n = train_images[n*128:(n+1)*128]
batch_n.shape

____________________________________________________________________________________________________________________

import numpy as np

x = np.array([1,4,6,9])
x.ndim
_________________________________________
x.shape
______________________________________________
y = np.array([[1,4,6,9],[1,4,6,9]])
y.ndim
_________________________________________________

p = np.array([[[1,4,6,9],
                [1,4,6,9]],
                [[1,4,6,9],
                [1,4,6,9]]
              ])
p.ndim

______________________________________________________________________________

x = np.array([[1,2,3],
              [4,5,6]])
print(x)
____________________________________________________________________________________
x.transpose()

_____________________________________________________________________________________
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
_______________________________________________________________________________________________

from pickle import load
iris = load_iris()
x = iris.data
y = iris.target

_____________________________________________________________________________________________
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
_______________________________________________________


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data = {
    'Age': [25, 30, np.nan, 22, 35],
    'Income': [50000, 60000, 55000, np.nan, 70000],
    'Gender': ['M', 'F', 'M', 'F', 'M'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Orignal Data:")
print(df)

#MISSING VALUES
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Income']] = imputer.fit_transform(df[['Age', 'Income']])
print("\nData After Imputation:")
print(df)

#eNCODING
le  = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
print("\nData After Label Encoding:")
print(df)


#One hit encoding
df = pd.get_dummies(df, columns=['Gender'])
print("\nData After One-Hot Encoding:")
print(df)

#min - max scaling
min_max_scaler = MinMaxScaler()
df[['Age', 'Income']] = min_max_scaler.fit_transform(df[['Age', 'Income']])
print("\nData After Min-Max Scaling:")
print(df) 


scaler = StandardScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])
print("\nData After Standard Scaling:")
print(df)

___________________________________________________________________________________________________________
scaler = StandardScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])
print("\nData After Standard Scaling:")
print(df)
____________________________________________________________________________________________________________
y_encoded = to_categorical(y)
print("\nOne-Hot Encoded Labels:")
print(y_encoded)
________________________________________________________________________________________________________

