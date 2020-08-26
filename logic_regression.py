import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def partition_data():
    """

    :param data_path:
    :return:
    """
    df = pd.read_csv((r"spam_data.csv"))
    labels = df['57'].to_numpy()
    df = df.drop(labels='57', axis=1).to_numpy()
    random = numpy.random.choice(len(df), len(df), replace=False)
    train_data_samples = df[random[1000:]]
    train_data_labels = labels[random[1000:]]
    test_data_samples = df[random[:1000]]
    test_data_labels = labels[random[:1000]]
    return train_data_samples, train_data_labels, test_data_samples, test_data_labels


def question_7_b():
    """

    :return:
    """
    x_train , y_train_labels, x_test, y_test_labels = partition_data()

    logistic_r = LogisticRegression()
    test_size = len(x_test)
    logistic_r.fit(x_train, y_train_labels)
    predict_proba = logistic_r.predict_proba(x_test)

    predict_1_values = numpy.argsort(predict_proba[:,1])[::-1]

    np = int(numpy.sum(y_test_labels))
    nn = len(y_test_labels) - np
    n_i = [0] * (np+1)

    index = 0
    for i in range(1,np):
        for j in range(index, test_size):
            if (y_test_labels[predict_1_values[j]] == 1):
                index = j+1
                break
        n_i[i] = index

    tpr = []
    fpr = []
    for i in range(len(n_i)):
        tpr.append(i/np)
        fpr.append(abs(n_i[i]- i)/ nn)

    return tpr, fpr


if __name__ == '__main__':

    for i in range(10):
        TPR, FPR = question_7_b()
        plt.plot(FPR, TPR)
    plt.title("ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


