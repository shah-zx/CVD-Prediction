Predicting-Heart-Disease :

IDE Used : VS Code 
Language : Python - 3.11.1
Lines of code - Approx - 50



Code Report :

First, we import the necessary packages: pandas for data handling,    train_test_split and GridSearchCV from scikit-learn for data splitting and hyperparameter tuning, KNeighborsClassifier for the classification model, StandardScaler for feature normalization, and SelectKBest and f_classif for feature selection.

Next, we load the heart disease dataset into a Pandas DataFrame using pd.read_csv(). The dataset contains 350 rows and 15 columns, including the target variable 'CVD Diagnosis', which is our binary classification label.

We separate the input features (X) and the target variable (y) using the drop() method and assigning the 'CVD Diagnosis' column to y.

We normalize the input features using StandardScaler to ensure that each feature has a mean of 0 and a standard deviation of 1. This is important because KNN is a distance-based algorithm and features with different scales can bias the model towards certain features.

We perform feature selection using SelectKBest with f_classif score function. SelectKBest selects the top k features with the highest ANOVA F-value, which measures the difference in means between two classes relative to the variance within each class.

We split the data into training and testing sets using train_test_split() with a test size of 0.9 and a random state of 128. This ensures that the data is split in a reproducible way.

We define a range of hyperparameters to search over using a dictionary. In this case, we search over the number of neighbors, the weight function used in prediction, and the power parameter for the Minkowski distance metric.

We initialize the KNeighborsClassifier model.

We perform a grid search over the hyperparameter space using cross-validation to find the best hyperparameters that maximize the accuracy of the model. In this case, we use a 5-fold cross-validation strategy.

We print the best hyperparameters and their corresponding score.

We use the best hyperparameters to train the KNeighborsClassifier on the full training set using grid_search.best_estimator_.

We evaluate the accuracy of the best KNeighborsClassifier on the testing data using best_knn.score().

We print the accuracy of the best KNeighborsClassifier on the testing data.


 Worked on increasing the accuracy by means :

Feature selection: The current code uses SelectKBest with f_classif score function to perform feature selection and select the top 10 features. However, selecting a different number of features or using a different score function could potentially improve the model's accuracy.

Hyperparameters of the KNeighborsClassifier: The code performs a grid search over the hyperparameters n_neighbors, weights, and p of the KNeighborsClassifier. However, there are other hyperparameters that could be tuned, such as algorithm and leaf_size, that could potentially improve the model's accuracy.

Random state: The code uses a fixed random state of 128 to split the data into training and testing sets. However, changing the random state could potentially result in different training and testing sets, which could impact the model's accuracy.

Test size: The code uses a test size of 0.9 to split the data into training and testing sets. However, using a different test size could potentially impact the model's accuracy.

Dataset: The current code uses a specific dataset 'Dataset_HD.csv' to train and test the model. However, using a different dataset with more relevant features or a larger sample size could potentially improve the model's accuracy.

Different terms - 

KNN (K-Nearest Neighbors) is a non-parametric supervised machine learning algorithm used for classification and regression tasks. Given a new observation or data point, the KNN algorithm classifies it based on the class of the K-nearest data points in the training set.

In the case of classification, the KNN algorithm calculates the distance between the new observation and all the training data points, then selects the K nearest data points and assigns the new observation to the class that appears most frequently among those K neighbors. The distance metric used can be Euclidean distance, Manhattan distance, or other distance measures.

In the case of regression, the KNN algorithm calculates the mean or median value of the K nearest data points and assigns it as the predicted value for the new observation.

The value of K is a hyperparameter that can be tuned to achieve better performance. In general, a larger value of K reduces the impact of noise in the data, but may also lead to oversmoothing of the decision boundary.

It seems that the reason why the CVD Diagnosis is taken as the target variable in the code is that the goal is to build a classification model to predict the risk of Cardiovascular Disease (CVD) based on the input features.

The input features are used to train the KNN classifier, which then predicts the target variable (CVD Diagnosis) for new data points. The CVD Diagnosis variable has two possible classes: 0 (indicating absence of CVD) and 1 (indicating presence of CVD).

The use of CVD Diagnosis as the target variable is relevant for this specific application, as the goal is to predict the presence or absence of CVD based on the input features. However, it's important to note that the choice of target variable may depend on the specific problem and the goal of the analysis.
