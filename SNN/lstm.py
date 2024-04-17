import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,BatchNormalization
# from tensorflow.keras.optimizers import SGD
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.keras.layers import Conv2D,MaxPooling2D,Conv1D,MaxPooling1D,GlobalAveragePooling1D
# from tensorflow.keras.layers import ReLU
# from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用CPU
# 提醒用户输入epoch大小
num_epochs = 150

# 将输入的字符串转换为整数
# num_epochs = int(num_epochs)

trainn = pd.read_csv('/mnt/data/CCM/snndatabase/dataset_ob17_31_overlap.csv', header=None)
label = trainn[4000].values
# label = trainn[0].values
train = trainn.drop([4000], axis=1)
y = label
x = train.values

# x=preprocessing.scale(x)#对x的每一列进行标准化

xt = preprocessing.scale(x.T)  # 对x的每一行进行标准化,每一条样本进行标准化
xt = xt.T
np.random.seed(1111)
idx = np.random.permutation(len(x))
# x1 = x[idx]
xt1 = xt[idx]
y1 = y[idx]

# x_train = x1[0:int(len(x)/10*7),:]
xt_train = xt1[0:int(len(x) / 10 * 7), :]
y_train = y1[0:int(len(x) / 10 * 7), ]

# x_test = x1[int(len(x)/10*7):len(x), :]
xt_test = xt1[int(len(x) / 10 * 7):len(x), :]
y_test = y1[int(len(x) / 10 * 7):len(x), ]

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
xt_train = xt_train.reshape((xt_train.shape[0], xt_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
xt_test = xt_test.reshape((xt_test.shape[0], xt_test.shape[1], 1))
num_classes = len(np.unique(y_train))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding="same", kernel_regularizer=l2(0.001))(
        input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    # conv1 = keras.layers.MaxPooling1D()(conv1)
    # conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same", kernel_regularizer=l2(0.001))(
    #     conv1)
    # conv2 = keras.layers.BatchNormalization()(conv2)
    # conv2 = keras.layers.ReLU()(conv2)
    # conv1 = keras.layers.BatchNormalization()(conv1)
    # conv1 = keras.layers.ReLU()(conv1)

    # Flatten = keras.layers.Flatten()(conv1)
    gap = keras.layers.GlobalAveragePooling1D()(conv1)

    dense1 = keras.layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001))(gap)
    # dense1 = keras.layers.BatchNormalization()(dense1)
    # dense1 = keras.layers.Dropout(0.2)(dense1)
    dense2 = keras.layers.Dense(64, activation="relu")(dense1)
    # dense2 = keras.layers.Dropout(0.2)(dense2)
    # dense2 = keras.layers.Dense(16, activation="relu")(dense1)
    dense3 = keras.layers.Dense(32, activation="relu")(dense2)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense3)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=xt_train.shape[1:])


# model = Sequential()
# model.add(LSTM(units=64, input_shape=(xt_train.shape[1], xt_train.shape[2]), return_sequences=True))
# model.add(LSTM(64, input_shape=(4000, 1), return_sequences=True))
# model.add(LSTM(32, return_sequences=False))

# model.add(LSTM(64, input_shape=(4000, 1)))
# model.add(Dense(units=num_classes, activation='softmax'))
# epochs = 350
# batch_size = 64

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.5,
                                              patience=20,
                                              verbose=1,
                                              mode='auto',
                                              min_delta=0.01,
                                              cooldown=0,
                                              min_lr=0.000005)

model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    xt_train,
    y_train,
    batch_size=64,
    epochs=num_epochs,
    callbacks=reduce_lr,
    validation_split=0.3,
    shuffle=True,
    verbose=1,
)

# model.save('lstm.h5')

# 获取当前日期和时间
now = datetime.datetime.now()
print(now)
# 将datetime对象格式化为字符串
model_now = now.strftime('%Y-%m-%d %H:%M:%S')
print(model_now)
model_name = model_now.replace("-", "_").replace(" ", "_").replace(":", "_")
# lstm_type = input("请输入lstm模型名字：")
# modelname = './models/'+lstm_type + '_' + str(num_epochs) + '_' + model_name + '.h5'
# imagename = './images/'+lstm_type + '_' + str(num_epochs) + '_' + model_name + '.h5'
# model.save(modelname)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim(0.3, 1)
# plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim(0, 2)
plt.legend()

plt.suptitle('column_normalazition_lstm')
# plt.savefig(imagename+'.jpg')
plt.show()
