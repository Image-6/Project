
#Load saved model and test on images.
#colorize_autoencoder300.model is trained on 300 epochs
#
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model(
    'C:/Users/karan/OneDrive/Desktop/Test/Output/colorize_autoencoder.model',
    custom_objects=None,
    compile=True)
img1_color=[]
img1=img_to_array(load_img('C:/Users/karan/OneDrive/Desktop/Test/Test/a.jpg'))
img1 = resize(img1 ,(256,256))
img1_color.append(img1)
img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))
output1 = model.predict(img1_color)
output1 = output1*128
result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
imsave("C:/Users/karan/OneDrive/Desktop/Test/Result/result.jpeg", lab2rgb(result))
