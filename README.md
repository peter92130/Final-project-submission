# Final-project-submission - Face Recognizer System

## 組員
    410421215 揚以德
    410421226 李冠穎
    410421241 涂耕熏

## 使用工具
* 程式語言 - Python
* 插件 - Keras , PIL
* Model - Sequential
* Batch size - 10
* Optimizer - RMSProp

## model
```

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
```

## 訓練方法
* 從13張照片中抽出2個當作測試目標來測試辨識率，將所有照片丟入以進行訓練，觀察其辨識率，然後儲存model
* 利用predict將訓練完的model讀入與測資，進行辨識並計算正確率

## 問題與解決方法
* 一開始想用GPU進行，不過我們對於環境安裝方式並不是太了解，花了一些時間也搞不定，最後決定以CPU進行，所以將epochs、batch_size調低
* 在訓練的方式與編譯上的問題，透過與其他組別的討論，才得以解決

## 辨識方式
***
|辨識準確      |辨識不準確    |成果    |
|:-----------:|:-----------:|:------:|
||||
