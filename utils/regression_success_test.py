import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def read_data(file_name):
    data = pd.read_excel(file_name)
    return data

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    loss_func = tf.keras.losses.MeanAbsoluteError()
    model.compile(optimizer=optimizer, loss=loss_func)
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=100, verbose=0)
    return model

def save_model(model, model_name):
    model.save(f"{model_name}.h5")

data = read_data('test.xlsx')

num_models = 6
models = [create_model() for _ in range(num_models)]
x_train_list = [np.array(data.iloc[:, 3*i]).reshape(-1, 1) for i in range(num_models)] #train data
y_train_list = [np.array(data.iloc[:, 3*i+1]).reshape(-1, 1) for i in range(num_models)] #train label

fig = plt.figure(figsize=(12, 8))

for i, model in enumerate(models):
    x_train, y_train = x_train_list[i], y_train_list[i]
    loss = 10
    epoch=0
    while loss >= 9.5 or loss<0:
        model = train_model(model, x_train, y_train)
        x_test = np.array(data.iloc[:, 3*i+2]).reshape(-1, 1)  # test data
        pred = model.predict(x_test)
        pred = pred.reshape(-1,)
        loss = np.mean(np.abs(pred - data.iloc[:, 3*i+1]) / pred) * 100
        print(f"Model{i+1}'s loss(%): {loss}")
        if epoch > 1000:
            break
        epoch+=1
    save_model(model, f"model_{i+1}")
    ax = fig.add_subplot(2, 3, i + 1)
    ax.scatter(x_train, y_train, s=30, c='b', marker='o', label='Train data')
    ax.scatter(x_test, pred, s=30, c='r', marker='x', label='Test data')
    if i==0:
        ax.set_title(f'CPF regression')
    elif i==1:
        ax.set_title(f'KD regression')
    elif i==2:
        ax.set_title(f'CPT regression')
    elif i==3:
        ax.set_title(f'FW regression')
    elif i==4:
        ax.set_title(f'CPS regression')
    elif i==5:
        ax.set_title(f'MG regrssion')
    ax.legend()
plt.show()

