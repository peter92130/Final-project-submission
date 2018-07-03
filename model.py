import win_unicode_console
win_unicode_console.enable()

from PIL import Image
import os
import numpy as np
import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

size_x = 180
size_y = 240

x_train = []
x_one_hot = []

y_test = []
y_one_hot = []


n=0
path=r".\Face Database"

for filename in os.listdir(path):
    if(filename[0] == 's'):
        test = Image.open(path+'\\'+filename)
        
        num = int(filename[1]) * 10 + int(filename[2]) 
        matrix = np.array(test) / 255
        num_temp = int(filename[4]) * 10 + int(filename[5]) 
        #print(matrix[0][0])
        
        if(n!=num):
            n=num
            random_list=[1,2,3,4,6,7,8,10,11,12,13,14,15]
            random.shuffle(random_list)
        if(num_temp==random_list[0] or num_temp==random_list[1]):
            y_test.append(matrix)
            y_one_hot.append(num)
            
        x_train.append(matrix)
        x_one_hot.append(num)
        
  
x_data = np.array(x_train)
x_label = np.array(x_one_hot)          

y_data = np.array(y_test)
y_label = np.array(y_one_hot)

print(x_data.shape)
print(y_data.shape)

x_label = to_categorical(x_label,num_classes = 51)
y_label = to_categorical(y_label,num_classes = 51)

model = Sequential()

model.add(Conv2D(128, 3, activation="relu", input_shape=(size_y, size_x, 3),padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3 , activation="relu",padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, activation="relu",padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, 3, activation="relu",padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(51, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

train_history = model.fit(x_data,x_label,
          batch_size=10,
          epochs=10,
          verbose=1,
          shuffle=True,
          validation_data=(y_data,y_label))

model.save(r'.\model.h5')

score = model.evaluate(y_data,y_label,verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
