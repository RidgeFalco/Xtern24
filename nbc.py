
# Note, this code is modified from a homework assignment I worked on during my Machine Learning course.

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

class NaiveBayes:
    """
    This class implements the Naive Bayes classifier.

    Attributes:
        alpha (float): The smoothing parameter for Laplace smoothing. Defaults
            to 1.0.
        n_features (int): The number of features in the training data. Initialized
            to `None`.
        class_labels (np.ndarray): The unique class labels in the training data.
            Initialized to `None`.
        class_probabilities (dict): The prior probability of each class. Initialized
            to `None`.
        feature_probabilities (list): The conditional probability of each feature
            given each class. Initialized to `None`.
                
    """
    def __init__(self, alpha: float = 1.0) -> None:
        """
        The constructor for NaiveBayes class. Initializes the smoothing parameter
        to `alpha`.

        Parameters:
            alpha (float): The smoothing parameter for Laplace smoothing. Defaults
                to 1.0.

        Returns:
            None
        """
        self.alpha = alpha
        self.n_features = None
        self.class_labels = None
        self.class_probabilities = None
        self.feature_probabilities = None

    def compute_class_probabilities(self, y_train: np.ndarray) -> Dict[Any, float]:
        """
        Computes the prior probability of each class.

        Parameters:
            y_train (np.ndarray): The class labels for the training data.

        Returns:
            class_probabilities (dict): The prior probability of each class.
                Each key is a class label and each value is the prior probability
                of that class in y_train.

        """
        classes, counts = np.unique(y_train, return_counts=True)
        
        class_probabilities = {}
        # Get total number of samples
        num_samples = y_train.size

        # Get total number of classes for Laplace Smoothing
        num_classes = classes.size

        # Iterate over each label and calculate the probabilty with Laplace Smoothing
        i = 0
        for c in classes:
            class_probabilities[c] = (counts[i] + self.alpha) / (num_samples + (num_classes * self.alpha))
            i = i + 1
        
        return class_probabilities
    
    def compute_feature_probabilities(self, X_j_train: np.ndarray, y_train: np.ndarray) -> Dict[Tuple[Any, Any], float]:
        """
        Computes the conditional probability of each feature given each class.

        Parameters:
            X_j_train (np.ndarray): A 1D array of the values of a given feature
                for all training examples.
            y_train (np.ndarray): The class labels for all training examples.

        Returns:
            feature_probabilities (dict): The conditional probability of each feature
                given each class. Each key is a tuple of the form (feature_value, class_label)
                and each value is the conditional probability of that feature value given
                that class. i.e:
                    feature_probabilities[(feature_value, class_label)] = P(X_j = feature_value | Y = class_label)
        """
        feature_probabilities = {}

        class_labels = np.unique(list(y_train))
        unique_features = set(X_j_train)

        # If 'NA' is not in the unique features, add it
        if 'NA' not in unique_features:
            # >>> YOUR CODE HERE >>>
            unique_features.add("NA")
            # <<< END OF YOUR CODE <<<

        # For each class, compute the conditional probability of each feature value
        # given that class
        for c in class_labels:
            for feature in unique_features:
                # Create key for dictionary
                key = (feature, c)
                
                # Get the joint count and marginal count
                count_joint = 0
                count_class = 0
                for i in range(y_train.size):
                    if X_j_train[i] == feature and y_train[i] == c:
                        count_joint = count_joint + 1
                    if y_train[i] == c:
                        count_class = count_class + 1

                # Using the joint count and marginal count, get the joint and marginal prob and add that value to the dict
                feature_probabilities[key] = (count_joint + self.alpha) / (count_class + (len(unique_features) * self.alpha))

        return feature_probabilities
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits a Naive Bayes model to the training data.

        Parameters:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The class labels for the training data.

        Returns:
            None
        """

        _, n_features = X_train.shape
        # Store the number of features
        self.n_features = n_features
        # Store the class labels
        self.class_labels = np.unique(list(y_train))
        # Compute the prior probability of each class

        # self.class_probabilities[class_label] = P(Y = class_label)
        self.class_probabilities = self.compute_class_probabilities(y_train)

        # self.feature_probabilities[i][(feature_value, class_label)] = P(X_i = feature_value | Y = class_label)
        # This line intilizes the array of dictionaries
        self.feature_probabilities = [{} for _ in range(n_features)]

        # This for loop calculates the feature probabilities for every feature in X_train
        for column in range(n_features):
            self.feature_probabilities[column] = self.compute_feature_probabilities(X_train[:, column], y_train)

    def predict_probabilities(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each class for each test example.

        Parameters:
            X_test (np.ndarray): The test data.

        Returns:
            np.ndarray: The predicted probabilities for each class for each test example.

        """

        assert X_test.shape[1] == self.n_features, "Number of features in X_test must match number of features in X_train"
        assert self.class_probabilities is not None, "Model has not been fit yet"
        assert self.feature_probabilities is not None, "Model has not been fit yet"
        assert self.class_labels is not None, "Model has not been fit yet"
        assert self.n_features is not None, "Model has not been fit yet"

        # Create a matrix of zeros with the correct shape
        probabilities = np.zeros((X_test.shape[0], len(self.class_labels)))
        # For each test example, compute the probability of each class
        for i in range(X_test.shape[0]):
            # For each class, compute the probability of the test example
            for c, label in enumerate(self.class_labels):
                # Store the probability in the correct position in the matrix
                probabilities[i, c] = self.class_probabilities[label]
                # For each feature, compute the probability of the test example
                for j in range(self.n_features):
                    # Store the probability in the correct position in the matrix
                    # If the feature value is not in the training data, use the
                    # probability of NA
                    if (X_test[i, j], label) not in self.feature_probabilities[j]:
                        probabilities[i, c] = probabilities[i, c] * self.feature_probabilities[j]["NA", label]
                    else:
                        # If the feature value is in the training data, use the
                        # probability of the feature value
                        probabilities[i, c] = probabilities[i, c] * self.feature_probabilities[j][X_test[i, j], label]

        # Normalize the probabilities
        for i, row in enumerate(probabilities):
            # Get total probabilty for that row
            total = 0
            for c in range(self.class_labels.size):
                total = total + row[c]

            # Final probabilty is the proportion of current prob over total prob
            for c in range(self.class_labels.size):
                probabilities[i, c] = probabilities[i, c] / total
        return probabilities

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Predicts the class for each test example.

        Parameters:
            probabilities (np.ndarray): The predicted probabilities for each
                class for each test example.

        Returns:
            np.ndarray: The predicted class for each test example.
        """

        assert self.class_labels is not None, "Model has not been fit yet"

        y_pred = []

        # Go over every test example and pick the label with the highest prob
        for row in probabilities:
            max_prob = np.argmax(row)
            y_pred.append(self.class_labels[max_prob])
        return np.array(y_pred)
    
    def evaluate(self, y_test: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the model on the test data. Computes the zero-one loss and
        squared loss.

        Parameters:
            y_test (np.ndarray): The true class labels for the test data.
            probabilities (np.ndarray): The predicted probabilities for each
                class for each test example.

        Returns:
            Tuple[float, float]: The zero-one loss and squared loss.
        """

        zero_one_loss = 0
        squared_loss = 0

        y_pred = self.predict(probabilities)

        # Code to get the zero-one loss
        i = 0
        incorrect = 0
        for label in y_pred:
            if label != y_test[i]:
                incorrect = incorrect + 1
            i = i + 1

        zero_one_loss = incorrect / y_test.size

        # Code to get the squared loss
        i = 0
        for label in y_test:
            # Make sure the label is an int since we will use it for getting the position its in in class_labels
            label = int(label)
            # Get index for the prob value we need to use and make sure it is an int so we can index on probabilities
            prob_index = int(self.class_labels[label])
            squared_loss = squared_loss + np.square(1 - probabilities[i, prob_index])
            i = i + 1

        # Now multiply the sum by 1/m where m is the number of test examples
        squared_loss = squared_loss / y_test.size

        return zero_one_loss, squared_loss