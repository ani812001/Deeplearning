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
