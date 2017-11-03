
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.image as mpimg
import os
import pandas as pd
from sklearn.utils import shuffle

def generator(data, batch_size=50):
    num_samples = len(data)
    # continue to run, yielding batches of images for model to train or validate.
    now = True
    while now:
        data = shuffle(data)
        # after shuffling data set, index is out of order - reset.
        data.reset_index(drop=True, inplace=True)
        for offset in range(0, num_samples, batch_size):
            # get the corect number of the last batch of images in dataset. 
            if (offset + batch_size) > num_samples:
                batch_size = num_samples - offset
            images = []
            # get all images in batch in a list.
            for i in range(batch_size):
                images.append(mpimg.imread(data['center'][offset + i]))
            flipped_images = []
            # flip all the images and get these in a list too.
            for i in images:
                flipped_images.append(np.fliplr(i))
            # put the two image lists together.
            all_images = images + flipped_images
            # convert the list to a numpy array.
            x = np.array(all_images)
            # get the steering angles for the images in the batch.
            steering_angle = data['steering'].values[offset:offset+batch_size]
            # get steering angles for the flipped images in batch.
            steering_angle_flipped = data['steering'].values[offset:offset+batch_size] * -1.0
            # combine the two steering angle numpy sets into one.
            y = np.concatenate((steering_angle, steering_angle_flipped), axis=0)
            
            # trim image to only see section with road, focus on road.
            x = x[:,80:,:,:] 
            
            yield (x, y)
   
        


# In[2]:

#! /usr/bin/env python
 
from keras.models import load_model, Model, Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf
tf.python.control_flow_ops = tf

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def main(_):
    
    # Load data log created in simulator into pandas dataframe
    data = pd.read_csv('driving_log.csv')
    
    # rename columns for dataframe.
    data.columns = ['center','left','right','steering','throttle','brake','speed']
    
    # shuffle and split dataframe into training and validation sets
    data = shuffle(data)
    train, validate = train_test_split(data, test_size=.2)
        
    # compile and train the model using the generator function    
    train_generator = generator(train, batch_size=256)
    validation_generator = generator(validate, batch_size=256)
    
    # Model Architecture
    model = Sequential()
    # normalize on images - input_shape: height, width, color_channels.
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(80, 320, 3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    
    # compile model using the adam optimizer.
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # use fit_generator to feed training data into memory - to avoid overloading memory.
    # data sets length doubled by adding flipped images in generator.
    model.fit_generator(train_generator, samples_per_epoch=len(train)*2, nb_epoch=5, 
                        validation_data=validation_generator, nb_val_samples=len(validate)*2) 
    print("Saving model weights and configuration file.")
    model.save('model.h5')  
      
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
 


# In[ ]:




# In[ ]:



