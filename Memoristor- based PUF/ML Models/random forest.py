import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import argparse
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier  # Import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier  # Import AdaBoostClassifier
from sklearn.svm import SVC
import torch

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dl', type=int, default=100000, metavar='DL',
                        help='data length')

    ## file_name
    parser.add_argument('--c_file_name', type=str, default='25_bit_challenge.csv',
                        help='challenge file name')
    parser.add_argument('--c_file_col', type=str, default='16_bit_challenge.csv',
                        help='challenge file name')

    parser.add_argument('--r_file_name', type=str, default='binary_Response_16.csv',
                        help='response file name')

    parser.add_argument('--method', type=str, default='adaboost',
                        help='[random_forest, adaboost, bag_tree, svm]')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

print(torch.cuda.is_available())
# 读取数据
df1 = pd.read_csv('input.csv', header=None)
df2 = pd.read_csv('output.csv', header=None)

# 设置随机种子
np.random.seed(42)

# 划分数据集
X = df1.iloc[1:1000001, :41] #
Y = df2.iloc[2:1000002, :]
print(Y)
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2,
random_state=42)
print(train_features.shape)
protion = 0.125
num = train_features.shape[0]
train_features = train_features[:int(protion * num)]

num2 = train_labels.shape[0]
train_labels = train_labels[:int(protion * num2)]
print(train_labels.shape)

if options.method == "random_forest":
    # Use RandomForestClassifier with 100 trees instead of SVC
    train_features, train_labels, test_features, test_labels = \
        train_features.astype('int'), train_labels.astype('int'), test_features.astype('int'), test_labels.astype('int')
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(train_features, train_labels.values.ravel())

    train_predictions = rf_model.predict(train_features)
    test_predictions = rf_model.predict(test_features)

    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

elif options.method == "adaboost":
    train_features, train_labels, test_features, test_labels = \
        train_features.astype('int'), train_labels.astype('int'), test_features.astype('int'), test_labels.astype('int')
    # Use AdaBoostClassifier instead of SVC
    ada_model = AdaBoostClassifier(random_state=42)
    ada_model.fit(train_features, train_labels.values.ravel())

    train_predictions = ada_model.predict(train_features)
    test_predictions = ada_model.predict(test_features)

    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

