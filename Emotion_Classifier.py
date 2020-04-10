from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 5 # Angry, sad, neutral, happy, surprise
img_rows,img_cols = 48,48 # input size of image for training model 
batch_size = 32 


# Directories of train and test set
train_data_dir = 'F:/My Projects/Emotion detector using keras and openCV/train'
validation_data_dir = 'F:/My Projects/Emotion detector using keras and openCV/validation'


#For increasing number of train data
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

#For increasing number of test data

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


# Making CNN

model = Sequential()

# Adding first layer
# It includes two convolutional layer and then max pooling

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1), activation = 'elu'))
# 32 is number of feature detector matrix and 3x3 is the sie of the feature detector matrix 
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1), activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # max pooling 2d as we are dealing with images
model.add(Dropout(0.2)) # It deativates 20% of neuron randomly to prevemt overfitting

#  Adding 2nd layer

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # max pooling 2d as we are dealing with images
model.add(Dropout(0.2)) # It deativates 20% of neuron randomly to prevemt overfitting



#  Adding 3rd layer

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # max pooling 2d as we are dealing with images
model.add(Dropout(0.2)) # It deativates 20% of neuron randomly to prevemt overfitting

#  Adding 4th layer

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # max pooling 2d as we are dealing with images
model.add(Dropout(0.2)) # It deativates 20% of neuron randomly to prevemt overfitting

# Flattening

model.add(Flatten())


# Full connection layer / Hidden layer

model.add(Dense(output_dim=64,kernel_initializer='he_normal', activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Adding another hidden layer

model.add(Dense(output_dim=64,kernel_initializer='he_normal', activation = 'elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Output Layer

model.add(Dense(num_classes,kernel_initializer='he_normal', activation = 'softmax'))


print(model.summary())


# For exporting trained dataset 

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# TO checkpoint 

checkpoint = ModelCheckpoint('Emotion_trained.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

# If model is not improving after 10 epochs then stop the training 

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          restore_best_weights=True
                          )
# to reducd learing rate if no decrement in monuitor after 3 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)


# All criteria for model training

callback = [earlystop,checkpoint,reduce_lr]

# Compiling CNN

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

train_samples = 24176
validation_samples = 3006


trained_model = model.fit_generator(
                train_generator, # training set
                steps_per_epoch=train_samples//batch_size, # number of images per epoch # Floor Division
                epochs=25, # Number of epochs
                validation_data=validation_generator, # test set
                validation_steps=validation_samples//batch_size, # number of images per epoch in testing
                callbacks=callback)  # Including callbacks























































