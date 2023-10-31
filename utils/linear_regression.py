import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def read_data(file_name):
    data = pd.read_excel(file_name)
    return data


data = read_data('reg_mob_nx.xlsx') #can change reg_mob_od, reg_eff_nx, reg_eff_od
num_models = 7
x_train_list = [np.array(data.iloc[1:, 3*i], dtype=np.float32).reshape(-1, 1) for i in range(num_models)] # train data
y_train_list = [np.array(data.iloc[1:, 3*i+1], dtype=np.float32).reshape(-1, 1) for i in range(num_models)] # train label
x_test_list = [np.array(data.iloc[1:, 3*i+2], dtype=np.float32).reshape(-1, 1) for i in range(num_models)] # train label

for i in range(num_models):
    not_nan_index = ~np.isnan(x_train_list[i]) & ~np.isnan(y_train_list[i]) & ~np.isnan(x_test_list[i])
    x_train_list[i] = x_train_list[i][not_nan_index].reshape(-1, 1)
    y_train_list[i] = y_train_list[i][not_nan_index].reshape(-1, 1)
    x_test_list[i] = x_test_list[i][not_nan_index].reshape(-1, 1)

models = [LinearRegression() for _ in range(num_models)]
fig = plt.figure(figsize=(12, 12))
fig.suptitle('mob in nx', fontsize=20, fontweight='bold')

for i in range(num_models):
    model = models[i]
    model.fit(x_train_list[i], y_train_list[i])

    x_train = x_train_list[i]
    y_train = y_train_list[i]
    x_test = x_test_list[i]

    pred = model.predict(x_test)
    a = model.coef_[0][0]  
    b = model.intercept_[0]  
    ax = fig.add_subplot(3, 3, i + 1)
    ax.scatter(x_train, y_train, s=15, c='b', marker='o', label='train')
    ax.scatter(x_test, pred, s=15, c='r', marker='x', label='test')
    ax.plot(x_test, a * x_test + b, color='g', label='line') 
    if i==0:
        ax.set_title(f'CPF regression')
        ax.set_xlabel('input size')
        ax.set_ylabel('latency') 
    elif i==1:
        ax.set_title(f'KD regression')
        ax.set_xlabel('FLOPs')
        ax.set_ylabel('latency')
    elif i==2:
        ax.set_title(f'CPT regression')
        ax.set_xlabel('output size')
        ax.set_ylabel('latency')
    elif i==3:
        ax.set_title(f'FW regression')
        ax.set_xlabel('output size')
        ax.set_ylabel('latency')
    elif i==4:
        ax.set_title(f'CPS regression')
        ax.set_xlabel('subgraph"s input size')
        ax.set_ylabel('latency')
    elif i==5:
        ax.set_title(f'MG regression')
        ax.set_xlabel('subgraph"s input size')
        ax.set_ylabel('latency')
    elif i==6:
        ax.set_title(f'LVS regression')
        ax.set_xlabel('FLOPs')
        ax.set_ylabel('latency')
    a = round(a,9)
    b = round(b,5)
    equation = "y = " + str(a) + "x + " + str(b)
    print(equation)
    ax.text(0.95, 0.05, equation, ha='right', va='bottom', color='red',transform=ax.transAxes, fontsize=10)
    ax.legend()
plt.subplots_adjust(hspace=0.5)
plt.show()
