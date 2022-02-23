import numpy as np
import pandas as pd
import lstm_enc_dec
import torch
from sklearn.model_selection import train_test_split

df = pd.DataFrame(pd.read_csv("preprocessed1.csv"))
df.head()

df.info()

Y = df['M_RAIN_PERCENTAGE']
numpred = df['Num_Predictions'].max()
X = df[['M_TRACK_TEMPERATURE', 'M_TRACK_LENGTH','M_FORECAST_ACCURACY', 'M_AIR_TEMPERATURE','M_NUM_WEATHER_FORECAST_SAMPLES', 'M_TRACK_ID','M_SEASON_LINK_IDENTIFIER', 'M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE', 'M_WEATHER_FORECAST_SAMPLES_M_WEATHER','M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE','M_TRACK_TEMPERATURE_CHANGE','M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE','M_AIR_TEMPERATURE_CHANGE']]
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(y_train.shape)

x_torch_train = torch.from_numpy(X_train.to_numpy()).type(torch.Tensor)
x_torch_test = torch.from_numpy(X_test.to_numpy()).type(torch.Tensor)
y_torch_train = torch.from_numpy(y_train.to_numpy().reshape(-1,1)).type(torch.Tensor)
y_torch_test = torch.from_numpy(y_test.to_numpy()).type(torch.Tensor)
x_torch_train = torch.reshape(x_torch_train,(5,571565,x_torch_train.shape[1]))
y_torch_train = torch.reshape(y_torch_train,(5,571565,y_torch_train.shape[1]))
print(x_torch_train.shape)
print(y_torch_train.shape)

#device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = lstm_enc_dec.lstm_seq2seq(input_size = x_torch_train.shape[2], hidden_size = 7)
model.to(device)
x_torch_train.to(device)
x_torch_test.to(device)
y_torch_train.to(device)
y_torch_test.to(device)
loss = model.train_model(x_torch_train, y_torch_train, n_epochs = 50, target_len = 5, batch_size = 200, training_prediction = 'recursive', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

print(loss)

import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)