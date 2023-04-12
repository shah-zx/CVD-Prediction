import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv("cvd_dataset_with_rules_binary.csv")

# Define the input and output data
X = df[["Systolic BP", "Diastolic BP", "Total Cholesterol", "Triglycerides"]]
Y = df["CVD Diagnosis"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(4, 1)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Reshape the input data for the LSTM model
X_train = np.array(X_train).reshape((X_train.shape[0], 4, 1))
X_test = np.array(X_test).reshape((X_test.shape[0], 4, 1))

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=64,
                    validation_data=(X_test, Y_test), verbose=1)

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test accuracy: {:.2f}%".format(score[1]*100))
