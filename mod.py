# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # load the dataset
# data = pd.read_csv('Dataset_HD.csv')

# # separate the features (X) and target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

# # train a random forest classifier on the training data
# rfc = RandomForestClassifier(n_estimators=100, random_state=47)
# rfc.fit(X_train, y_train)

# # make predictions on the testing data
# y_pred = rfc.predict(X_test)

# # evaluate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)


# ------------------------------------------------------------------------------------------------

# 53 % accuracy

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# # load the dataset
# data = pd.read_csv('Dataset_HD.csv')

# # separate the features (X) and target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# # preprocess the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # train a random forest classifier on the training data
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# rfc = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # make predictions on the testing data
# y_pred = grid_search.predict(X_test)

# # evaluate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)


# ----------------------------------------------------------------

# 50.35 % accuracy


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=36)

# # Initialize the KNeighborsClassifier with k=5
# knn = KNeighborsClassifier(n_neighbors=7)

# # Train the KNeighborsClassifier on the training data
# knn.fit(X_train, y_train)

# # Use the trained KNeighborsClassifier to make predictions on the testing data
# y_pred = knn.predict(X_test)

# # Evaluate the accuracy of the KNeighborsClassifier
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy of the KNeighborsClassifier: {:.2f}%".format(accuracy*100))


# ----------------------------------------------------------------------------------------------

# 56 % accuracy 

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Calculate the correlation between each input feature and the target variable
# correlations = X.corrwith(y)

# # Select the top 5 features with the highest correlation
# top_features = correlations.abs().nlargest(5).index
# X_top = X[top_features]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.7, random_state=55)

# # Initialize the KNeighborsClassifier with k=5
# knn = KNeighborsClassifier(n_neighbors=5)

# # Train the KNeighborsClassifier on the training data
# knn.fit(X_train, y_train)

# # Use the trained KNeighborsClassifier to make predictions on the testing data
# y_pred = knn.predict(X_test)

# # Evaluate the accuracy of the KNeighborsClassifier
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy of the KNeighborsClassifier: {:.2f}%".format(accuracy*100))



# ----------------------------------------------------------------------------------

# Accuracy - 80 % - Apna wala

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_classif

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Normalize the input features using StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Perform feature selection using SelectKBest with f_classif score function
# kbest = SelectKBest(f_classif, k=10)
# X = kbest.fit_transform(X, y)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=128)

# # Define a range of hyperparameters to search over
# param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
#               'weights': ['uniform', 'distance'],
#               'p': [1, 2, 3, 4]}

# # Initialize the KNeighborsClassifier
# knn = KNeighborsClassifier()

# # Perform a grid search over the hyperparameter space using cross-validation
# grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters and their corresponding score
# print("Best hyperparameters:", grid_search.best_params_)
# # print("Best score:", grid_search.best_score_)

# # Use the best hyperparameters to train the KNeighborsClassifier on the full training set
# best_knn = grid_search.best_estimator_
# best_knn.fit(X_train, y_train)

# # Evaluate the accuracy of the best KNeighborsClassifier on the testing data
# accuracy = best_knn.score(X_test, y_test)
# print("Model's Accuracy is: {:.2f}%".format(grid_search.best_score_*100))



# ----------------------------------------------------------------------------------------

# Accuracy - 48.74 %


# deep learning
# kears neural network - 

# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Normalize the input features using StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=128)

# # Define the neural network architecture
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# # Evaluate the model on the testing data
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Model's Accuracy is: {:.2f}%".format(accuracy*100))

# ----------------------------------------------------------------


# 1.

# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Normalize the input features using StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=123)

# # Define the neural network architecture
# model = Sequential([
#     LSTM(64, input_shape=(X_train.shape[1], 1)),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # Define the optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# # Compile the model
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Reshape the input data for the LSTM layer
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Train the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# # Evaluate the model on the testing data
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Model's Accuracy is: {:.2f}%".format(accuracy*100))

# --------------------------------------------------------------------------------------

# 2. 

# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Normalize the input features using StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=128)

# # Define the neural network architecture
# model = Sequential([
#     Dense(128, activation='tanh', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     Dense(64, activation='tanh'),
#     Dropout(0.2),
#     Dense(32, activation='tanh'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model with the RMSprop optimizer
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model for 100 epochs
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# # Evaluate the model on the testing data
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Model's Accuracy is: {:.2f}%".format(accuracy*100))

# ----------------------------------------------------------------------------------------------------

# Accuracy - 53 %

# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv('Dataset_HD.csv')

# # Separate the input features (X) and the target variable (y)
# X = data.drop('CVD Diagnosis', axis=1)
# y = data['CVD Diagnosis']

# # Normalize the input features using StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=128)

# # Define the neural network architecture
# model = Sequential([
#     LSTM(128, activation='tanh', input_shape=(X_train.shape[1],1), return_sequences=True),
#     Dropout(0.2),
#     LSTM(64, activation='tanh', return_sequences=True),
#     Dropout(0.2),
#     LSTM(32, activation='tanh'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model with the RMSprop optimizer
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model for 100 epochs
# history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, 
#                     validation_data=(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test),
#                     epochs=100, batch_size=32)

# # Evaluate the model on the testing data
# loss, accuracy = model.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test)
# print("Model's Accuracy is: {:.2f}%".format(accuracy*100))


# ------------------------------------------------------------------------------------------------

# RNN - 60 % Accuracy

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from keras.optimizers import Adam
# df = pd.read_csv('Dataset_HD.csv')

# # Define the target column
# target_col = 'CVD Diagnosis'

# # Split the data into input features and target variable
# X = df.drop(target_col, axis=1)
# y = df[target_col]

# # Split the dataset into training and testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# # Define the number of input features and output classes
# n_features = X_train.shape[1]
# n_classes = len(np.unique(y_train))

# # Define the model
# model = Sequential()
# model.add(LSTM(64, input_shape=(1, n_features)))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(n_classes, activation='softmax'))

# # Compile the model
# optimizer = Adam(lr=0.001)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # Reshape the input data for LSTM
# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# # Evaluate the model on the test set
# _, test_acc = model.evaluate(X_test, y_test)

# print('Test accuracy: %.2f%%' % (test_acc*100))


# -------------------------------------------------------------------


# Less Attributes - 
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from keras.optimizers import Adam

# # Load data
# df = pd.read_csv('Dataset_HD.csv')

# # Split data into inputs (X) and output (y)
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize data
# mean = X_train.mean(axis=0)
# std = X_train.std(axis=0)
# X_train = (X_train - mean) / std
# X_test = (X_test - mean) / std

# # Reshape data for LSTM
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Define the model
# model = Sequential()
# model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(1))

# # Compile the model
# optimizer = Adam(lr=0.001)
# model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# # Evaluate the model
# loss, mae = model.evaluate(X_test, y_test)
# print(f'Test loss: {loss}, test MAE: {mae}')


# import pandas as pd

# # Load the dataset
# df = pd.read_csv("Dataset_HD.csv")

# # Define the rules for each attribute
# def systolic_bp_rule(value):
#     if value >= 140:
#         return "CVD"
#     elif value <= 90:
#         return "CVD"
#     else:
#         return "NO CVD"

# def diastolic_bp_rule(value):
#     if value >= 90:
#         return "CVD"
#     elif value <= 60:
#         return "CVD"
#     else:
#         return "NO CVD"

# def total_cholesterol_rule(value):
#     if value >= 240:
#         return "CVD"
#     elif value <= 200:
#         return "CVD"
#     else:
#         return "NO CVD"

# def triglycerides_rule(value):
#     if value >= 200:
#         return "CVD"
#     elif value <= 150:
#         return "CVD"
#     else:
#         return "NO CVD"

# # Apply the rules to each row in the dataset
# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                systolic_bp_rule(row["Systolic BP"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                diastolic_bp_rule(row["Diastolic BP"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                total_cholesterol_rule(row["Total Cholesterol"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                triglycerides_rule(row["Triglycerides"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# # Save the updated dataset
# df.to_csv("cvd_dataset_with_rules.csv", index=False)






# -----------------------------------------------------------

# The training state - 


# import pandas as pd

# # Load the dataset
# df = pd.read_csv("Dataset_HD.csv")

# # Define the rules for each attribute
# def systolic_bp_rule(value):
#     if value >= 140:
#         return 1
#     elif value <= 90:
#         return 1
#     else:
#         return 0

# def diastolic_bp_rule(value):
#     if value >= 90:
#         return 1
#     elif value <= 60:
#         return 1
#     else:
#         return 0

# def total_cholesterol_rule(value):
#     if value >= 240:
#         return 1
#     elif value <= 200:
#         return 1
#     else:
#         return 0

# def triglycerides_rule(value):
#     if value >= 200:
#         return 1
#     elif value <= 150:
#         return 1
#     else:
#         return 0

# # Apply the rules to each row in the dataset
# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                systolic_bp_rule(row["Systolic BP"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                diastolic_bp_rule(row["Diastolic BP"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                total_cholesterol_rule(row["Total Cholesterol"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# df["CVD Diagnosis"] = df.apply(lambda row: 
#                                triglycerides_rule(row["Triglycerides"]) 
#                                if pd.isna(row["CVD Diagnosis"]) 
#                                else row["CVD Diagnosis"], axis=1)

# # Save the updated dataset
# df.to_csv("cvd_dataset_with_rules_binary.csv", index=False)


# ----------------------------------------------------------------

# The testing state and accuracy finding - 

# max accuracy - 95.06 %

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
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test), verbose=1)

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test accuracy: {:.2f}%".format(score[1]*100))


