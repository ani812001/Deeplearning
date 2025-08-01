!pip install tensorflow

-----------------------------------

import tensorflow as  tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

------------------------------------------------------

!pip install keras==2.10.0

-------------------------------------------------------


!pip install protobuf==3.19.6

-----------------------------------------------------

import numpy as np
x = np.array(12)
# x is 0 Dimensional 

----------------------------------------------------

x = np.array([12,3,6,14])
print(x)
print(x.ndim)
print(x.shape)
# x is 1 Dimensional

-----------------------------------------------

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x)
print(x.ndim)
print(x.shape)
#x is 2 Dimensional

-------------------------------------------------------


import tensorflow as tf

#creating a constant
a = tf.constant([[1,2] , [3,4]], dtype=tf.int32)
b = tf.constant([[5,6] , [7,8]], dtype=tf.int32)

____________________________________________________________

print ("Addition:\n" , tf.add(a,b))
print ("Subtraction:\n" , tf.subtract(a,b))
print ("Multiplication:\n" , tf.multiply(a,b))
print ("Division:\n" , tf.divide(a,b))
____________________________________________________________

#3D array


import numpy as np

x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],

              [[8, 81, 5, 37, 3],
               [9, 82, 6, 38, 4],
               [10, 83, 7, 39, 5]]])

print(x)
print(x.ndim)   # Should output 3
print(x.shape)  # Should output (2, 3, 5)
# x is 3 Dimensional

____________________________________________________________________________

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

______________________________________________________________________________________________

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

_____________________________________________________________________________

digit = train_images[0]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

_______________________________________________________________________________

bottom_right_corner = train_images[:,14:, 14:]
plt.imshow(bottom_right_corner[0], cmap=plt.cm.binary)
plt.show()

_________________________________________________________________________________
center_patches = train_images[:, 7:-7, 7:-7]
plt.imshow(center_patches[0], cmap=plt.cm.binary)
plt.show()

_____________________________________________________________________________________

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train) , (x_test, y_test) = tf.keras.datasets.mnist.load_data()

_________________________________________________________________________________________

gray_scale = 255

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

print("Feature matrix (x_train):", x_train.shape)
print("Target vector (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target vector (y_test):", y_test.shape)
________________________________________________________________________


