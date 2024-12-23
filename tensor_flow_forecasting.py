import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(data, seq_length, predict_length):
    X, y = [], []   # input sequences (past values), and target (next) values

    for i in range(len(data)-seq_length-predict_length+1):
        X.append(data[i:i+seq_length])  # sequence of length 'seq_length'
        y.append(data[i+seq_length:i+seq_length+predict_length])    # value immediately after sequence
    return np.array(X), np.array(y)

# load the dataset
path = "D:/Pycharm Projects/Demos/skLearnDemo/btcusd_1-min_data.csv"
data = pd.read_csv(path)
data = data[::1440]

data.set_index('Timestamp', inplace=True)

# normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']])

# split the data into train and test sets
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


seq_length = 60 # number of times steps to look back (1 month)
predict_length = 30  # number of time steps to look forward
X_train, y_train = create_sequences(train_data, seq_length, predict_length)
X_test, y_test = create_sequences(test_data, seq_length, predict_length)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)


X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# defining and training the lstm model
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model = Sequential([
    LSTM(300, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    LSTM(300, activation='relu'),
    Dense(predict_length) #output 30 predictions
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=4096, verbose=1, callbacks=[early_stopping])

# ----------------------------------------------------
# prediction and visualization
# ----------------------------------------------------

# make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# inverse transform predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# plot predictions
plt.figure(figsize=(10, 6))

# plot the real data
plt.plot(data.index[seq_length:], data['Close'].iloc[seq_length:], label='Actual', color='blue', linewidth=1)

# plot training predictions
plt.plot(data.index[seq_length:seq_length+len(train_predictions)],
         train_predictions[:, 0], label='Train Predictions (First Step)', color='green', linewidth=1, linestyle='--')


# plot testing predictions
test_pred_index = range(seq_length+len(train_predictions),
                        seq_length+len(train_predictions)+len(test_predictions))
plt.plot(data.index[test_pred_index], test_predictions[:, 0], label='Test Predictions', color='orange', linewidth=1, linestyle=':')


# make the chart clearer
# make the chart clearer
plt.title('Bitcoin Price Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

