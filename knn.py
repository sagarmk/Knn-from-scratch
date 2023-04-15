import csv
import random
import math
import operator
import urllib.request
import numpy as np

class IrisDataset:
    """
    Class for loading and splitting the Iris dataset
    into training and test sets.

    Attributes:
    filename (str): filename of the Iris dataset
    split (float): ratio of the dataset to be used for training
    training_set (list): training set of the dataset
    test_set (list): test set of the dataset
    """        
    def __init__(self, filename, split):
        """
        Initializes the IrisDataset object with filename and split ratio.
        
        Args:
        filename (str): filename of the Iris dataset
        split (float): ratio of the dataset to be used for training
        """
        self.filename = filename
        self.split = split
        self.training_set = []
        self.test_set = []
    
    def load_dataset(self):
        """
        Loads the Iris dataset into training and test sets
        by splitting the dataset according to the given split ratio.
        """

        with urllib.request.urlopen(self.filename) as response:
            lines = csv.reader(response.read().decode('utf-8').splitlines())
            dataset = list(lines)
            for x in range(len(dataset) - 1):
                for y in range(4):
                    dataset[x][y] = float(dataset[x][y])

                if random.random() < self.split:
                    self.training_set.append(dataset[x])
                else:
                    self.test_set.append(dataset[x])
        
        self.training_set = np.array(self.training_set)
        self.test_set = np.array(self.test_set)

    

class EuclideanDistance:
    """
    Class for calculating the Euclidean distance between two instances.
    """
    @staticmethod
    def calculate(instance1, instance2, length):
        """
        Calculates the Euclidean distance between two instances.
        
        Args:
        instance1 (list): first instance
        instance2 (list): second instance
        length (int): length of the instances
        
        Returns:
        float: Euclidean distance between the two instances
        """
        distance = 0
        for x in range(length):
            distance += pow((float(instance1[x]) - float(instance2[x])), 2)

        return math.sqrt(distance)


class Neighbors:
    """
    Class for selecting k nearest neighbors of a given test instance.
    """
    @staticmethod
    def select(training_set, test_instance, k):
        """
        Selects k nearest neighbors of a given test instance.
        
        Args:
        training_set (list): list of training instances
        test_instance (list): test instance
        k (int): number of nearest neighbors to select
        
        Returns:
        list: k nearest neighbors of the test instance
        """
        distances = []
        length = len(test_instance) - 1
        for x in range(len(training_set)):
            dist = EuclideanDistance.calculate(test_instance, training_set[x], length)
            distances.append((training_set[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors


class Response:
    """This class calculates the class prediction of a test instance.

    Given a list of `neighbors` (neighbors are instances from the training set), 
    it calculates the majority class prediction using a voting mechanism. 
    """
    @staticmethod
    def predict(neighbors):
        """Calculates the class prediction of a test instance.

        Args:
        neighbors: list of instances from the training set.

        Returns:
        str: majority class prediction of a test instance.
        """
        class_votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]


class Accuracy:
    """This class calculates the accuracy of the classifier predictions.

    Given a `test_set` (actual class labels of the instances) and a list of 
    `predictions` (classifier's predictions), it calculates the accuracy of the 
    classifier's predictions.
    """
    @staticmethod
    def measure(test_set, predictions):
        """Calculates the accuracy of classifier's predictions.

        Args:
        test_set: list of instances from the test set with actual class labels.
        predictions: list of classifier's predictions for instances in the test set.

        Returns:
        float: accuracy of classifier's predictions.
        """
        correct = 0
        for x in range(len(test_set)):
            if test_set[x][-1] in predictions[x]: 
                correct = correct + 1
        return (correct / float(len(test_set)) * 100) 


def main(filename, split, k):
    # prepare data
    iris = IrisDataset(filename, split)
    iris.load_dataset()

    # generate predictions
    predictions = []
    results = []
    for x in range(len(iris.test_set)):
        neighbors = Neighbors.select(iris.training_set, iris.test_set[x], k)
        result = Response.predict(neighbors)
        predictions.append(result)
        results.append(f'> predicted={repr(result)}, actual={repr(iris.test_set[x][-1])}')

    accuracy = Accuracy.measure(iris.test_set, predictions)
    results.append(f'Accuracy: {repr(accuracy)}%')

    X_train = iris.training_set[:, :-1]
    y_train = iris.training_set[:, -1]
    X_test = iris.test_set[:, :-1]
    y_test = iris.test_set[:, -1]

    return X_train, y_train, X_test, y_test



    

if __name__ == "__main__":
    main()
