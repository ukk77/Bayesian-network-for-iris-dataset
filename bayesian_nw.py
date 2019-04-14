__author__ = "Uddesh Karda"
"FIS project 2 Bayesian network problem"

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.stats import norm
import random
import math


attributes = ["sepal_length","sepal_width","petal_length","petal_width"]
Classes = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
data = {}
train_data = []
test_data = []
data_list = []
count = {}

def main():
    """
    Main function initialises network classification
    :return:
    """
    random_baseline_count = 0
    k = int(input("Enter number of rows in test data : " + "\n"))
    splitData(k)
    loadData()
    summarize()
    correct_vals = 0
    for i in range(len(test_data)):
        data_line = (test_data[i]).split(',')
        result = classify(test_data[i])
        result = np.array(result)/np.sum(result)
        rand_class = random.randrange(0,len(Classes))
        if Classes[np.argmax(result)] == data_line[-1].rstrip():
            correct_vals += 1
        if Classes[rand_class] == data_line[-1].rstrip():
            random_baseline_count += 1
    print("random baseline = " + str((random_baseline_count/len(test_data))*100) + '%' + "\n")
    print("accuracy = " + str((correct_vals/len(test_data))*100) + '%' + "\n")
    max_count = 0
    max_key = []
    for key, value in count.items():
        if value >= max_count:
            max_key.append(key)
    print("Classes with majority frequency are as follows : ")
    for i in max_key:
        print(i, str(((count.get(i) / sum(list(count.values()))) * 100)) + '%')


def classify(line):
    """
    Takes a line as input and classifies the given line by using Bayesian network probabilities
    :param line: A line of data
    :return: Result class
    """
    result = []
    line_list = []
    if line == '\n':
        return
    line_list = line.split(',')
    for i in range(len(Classes)):
        prob = 1
        for j in range(len(attributes)):
            prob *= conditional_prob(float(line_list[j]),i,j)
        result.append(prob)
    return result


def conditional_prob(x,i,j):
    """
    Finds conditional probability
    :param x: Value from line of data to be classified
    :param i: It's attribute
    :param j: It's index
    :return: float probability value
    """

    return norm.pdf(x,((data.get(attributes[j])).get(Classes[i])).get('mean'),
                    ((data.get(attributes[j])).get(Classes[i])).get('var') )


def splitData(n):
    """
    Splits the data into training and testing
    :param n: Number of rows in test data set
    :return: -
    """
    for j in range(len(Classes)):
        count[Classes[j]] = 0
    random_list = []
    for i in range(1,math.floor(n/3)):
        random_list.append(random.randrange(0,50))
    for i in range(math.floor(n/3), math.floor(2*(n/3))):
        random_list.append(random.randrange(50,100))
    for i in range(math.floor(2*n/3),math.floor(n)):
        random_list.append(random.randrange(100,150))
    f = open('iris.data')
    line = f.readline()
    while line != "\n":
        data_list.append(line)
        line_vals = line.split(',')
        if  'Iris-setosa' in line:
            count['Iris-setosa'] = count.get('Iris-setosa') + 1
        elif 'Iris-versicolor' in line:
            count['Iris-versicolor'] = count.get('Iris-versicolor') + 1
        else:
            count['Iris-virginica'] = count.get('Iris-virginica') + 1
        line = f.readline()
    f.close()
    for i in range(len(data_list)):
        if data_list[i] == '\n':
            continue
        if i in random_list:
            train_data.append(data_list[i])
        else:
            test_data.append(data_list[i])


def mean(list_of_data):
    """
    Finds mean of given array
    :param list_of_data: list of data from dataset
    :return: mean of list of data
    """
    return np.mean(list_of_data)


def variation(list_of_data):
    """
    Finds variation of given array
    :param list_of_data: list of data form given dataset
    :return: variation of list of data
    """
    return np.var(list_of_data)


def summarize():
    """
    Puts the data, mean and variance for the data into a dictionary correctly
    :return: -
    """

    for i in range(len(attributes)):
        for j in range(len(Classes)):
            list_of_data = (data.get(attributes[i])).get(Classes[j])
            data[attributes[i]][Classes[j]] = {}
            data[attributes[i]][Classes[j]]['mean'] = mean(list_of_data)
            data[attributes[i]][Classes[j]]['var'] = variation(list_of_data)


def loadData():
    """
    Function to get data from file
    :return:-
    """
    for i in range(len(attributes)):
        data[attributes[i]] = {}
        for j in range(len(Classes)):
            data[attributes[i]][Classes[j]] = []
    for line in train_data:
        line_vals = line.split(',')
        if 'Iris-setosa' in line:
            data["sepal_length"]['Iris-setosa'].append(float(line_vals[0]))
            data["sepal_width"]['Iris-setosa'].append(float(line_vals[1]))
            data["petal_length"]['Iris-setosa'].append(float(line_vals[2]))
            data["petal_width"]['Iris-setosa'].append(float(line_vals[3]))

        elif 'Iris-versicolor' in line:
            data["sepal_length"]['Iris-versicolor'].append(float(line_vals[0]))
            data["sepal_width"]['Iris-versicolor'].append(float(line_vals[1]))
            data["petal_length"]['Iris-versicolor'].append(float(line_vals[2]))
            data["petal_width"]['Iris-versicolor'].append(float(line_vals[3]))

        else:
            data["sepal_length"]['Iris-virginica'].append(float(line_vals[0]))
            data["sepal_width"]['Iris-virginica'].append(float(line_vals[1]))
            data["petal_length"]['Iris-virginica'].append(float(line_vals[2]))
            data["petal_width"]['Iris-virginica'].append(float(line_vals[3]))


if __name__ == '__main__':
    main()