IDE Used : VS Code 
Language : Python - 3.11.1
Lines of code - Approx - 40
Accuracy - 95.06 % 

Code Report - 

Ranges of the Attributes in the dataset - 

Systolic BP - Actual Range - 120 - 140 mm Hg
                     Model’s Range - 100 - 150 mm Hg
Diastolic BP - Actual Range - 80 - 100 mm Hg
                     Model’s Range - 70 - 100 mm Hg
Total Cholestrol - Actual Range - 200 - 240 mm Hg
                            Model’s Range - 180 - 250 mm Hg
Triglycerides - Actual Range - 150 - 550 mg/dL (milligrams per deciliter)
                       Model’s Range - 130 - 550 mg/dL







Approach for building the Model  - 

Training Phase -

Code is written in Python and uses the Pandas library for data manipulation. It applies a set of rules to a dataset and updates a column called "CVD Diagnosis" with binary values (1 or 0) based on the outcome of the rules.

The dataset is loaded from a CSV file called "Dataset_HD.csv" using the read_csv() function of the Pandas library and is stored in a variable called "df".

Four functions are defined to implement the rules for each attribute: systolic_bp_rule(), diastolic_bp_rule(), total_cholesterol_rule(), and triglycerides_rule(). Each function takes a single parameter (value) and returns either 1 or 0 based on the value of the parameter and the rule defined in the function.

The apply() function of the Pandas library is used to apply the rules to each row of the dataset. Four separate apply() functions are called for each attribute, and the results are stored in the "CVD Diagnosis" column of the dataset.

In each apply() function, a lambda function is used to call the corresponding rule function (e.g., systolic_bp_rule()) with the value from the current row of the dataset. If the "CVD Diagnosis" column of the current row is empty (NaN), the rule is applied and the result is stored in the "CVD Diagnosis" column. Otherwise, the value from the "CVD Diagnosis" column is retained.

Finally, the updated dataset is saved to a CSV file called "cvd_dataset_with_rules_binary.csv" using the to_csv() function of the Pandas library with the index parameter set to False to avoid saving the index of the rows in the dataset.

Testing Phase - 

Code is written in Python and uses the Keras library (a high-level neural networks API) with TensorFlow backend for implementing a LSTM model to predict CVD diagnosis based on the input data.

The dataset is loaded from a CSV file called "cvd_dataset_with_rules_binary.csv" using the read_csv() function of the Pandas library and is stored in a variable called "df".

The input and output data are defined. X is a dataframe containing the columns "Systolic BP", "Diastolic BP", "Total Cholesterol", and "Triglycerides" of the dataset, and Y is a series containing the "CVD Diagnosis" column of the dataset.

The data is split into training and testing sets using the train_test_split() function of the scikit-learn library with a test size of 0.2, which means that 20% of the data is reserved for testing.

The LSTM model is defined using the Sequential() function of the Keras library. The model consists of three LSTM layers, each followed by a Dropout layer to prevent overfitting, and a Dense layer with a sigmoid activation function to produce the binary output. The input_shape parameter of the first LSTM layer is set to (4, 1) because there are four input features, and the model expects a 3D input with a batch size of None, four time steps, and one feature.

The model is compiled using the compile() function, which takes three parameters: the loss function, the optimizer, and a list of metrics to be evaluated during training and testing. The loss function is "binary_crossentropy", the optimizer is Adam with a learning rate of 0.001, and the metrics are "accuracy".

The input data for the LSTM model is reshaped using the reshape() function of the NumPy library, which converts the data from a 2D array to a 3D array with a batch size of None, four time steps, and one feature.

The model is trained using the fit() function, which takes four parameters: the input data, the output data, the number of epochs (iterations over the entire dataset), the batch size (number of samples per gradient update), and the validation data. The verbose parameter is set to 1 to display the progress during training. The training history is stored in a variable called "history".

The model is evaluated using the evaluate() function, which takes two parameters: the input data and the output data. The test accuracy is calculated as the second element of the score list returned by the evaluate() function, multiplied by 100, and printed to the console using the print() function with the format() method to display the result as a percentage with two decimal places.


 Parameters manipulated for increasing the accuracy - 
 
Number of LSTM layers: Increased the number of LSTM layers  helped in  the model capture more complex patterns in the data.

Number of LSTM units: Increased the number of LSTM units in each layer helped in  the model capture more fine-grained patterns in the data.

Dropout rate: Increased the dropout rate helped prevent overfitting, which occurs when the model performs well on the training data but poorly on new, unseen data.

Learning rate: Increased the learning rate helped the model converge faster during training, but if the learning rate is set too high, the model may fail to converge.

Batch size: Increased the batch size helped speed up training, but if the batch size is set too large, the model may not generalize well to new data.

Number of epochs: Increased the number of epochs which allowed the model to continue learning for longer and potentially improve its performance. However, if the number of epochs is set too high, the model may overfit to the training data.

Additional features: Adding additional relevant features to the input data  provided the model with more information to help it make better predictions.



Implementation - 

Made a Website using HTML , CSS , and JS. It is having following pages - 

Login Page 
Sign Up Page
Welcome Page 
CVD Prediction Page 
CVD Result Page
