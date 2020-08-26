import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from QDA import QDA
from LDA import LDA



def main():
    """

    :return:
    """
    spam_data = pd.read_csv("spam.data.txt", sep=" ", header=None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    x = spam_data.drop([57], axis=1)
    y = spam_data[57]

    lda_train_err_2 = 0
    qda_train_err_1 = 0
    qda_test_err_3 = 0
    lda_test_err_4 = 0

    for i in range(10):

        train_x, test_x, train_y, test_y = tts(x, y,test_size=1000)
        test_x = test_x.iloc[:, [1,4,21,20,10]]
        train_x = train_x.iloc[:, [1,4,21,20,10]]

        qda = QDA()
        qda.fit(train_x, train_y)

        lda = LDA()
        lda.fit(train_x, train_y)

        train_y = np.array(train_y)
        train_x = np.array(train_x)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        temp_counter = 0
        temp_counter2 = 0
        for index in range(len(train_x)):
            if qda.predict(train_x[index]) != train_y[index]:
                temp_counter += 1
            if lda.predict(train_x[index]) != train_y[index]:
                temp_counter2 += 1
        qda_train_err_1 +=  temp_counter / len(train_y)
        lda_train_err_2 +=  temp_counter2 / len(train_y)

        temp_counter = 0
        temp_counter2 = 0
        for index in range(len(test_x)):
            if qda.predict(test_x[index]) != test_y[index]:
                temp_counter += 1
            if lda.predict(test_x[index]) != test_y[index]:
                temp_counter2 += 1
        qda_test_err_3 +=  temp_counter / len(test_y)
        lda_test_err_4 +=  temp_counter2 / len(test_y)

    print("QDA errors:")
    print("test - %lf train -  %f " % (
        qda_test_err_3 / 10, qda_train_err_1 / 10))

    print("LDA errors:")
    print("test -  %lftrain -  %f" % (
        lda_test_err_4 / 10, lda_train_err_2 / 10))

    qda_eigen_vals = qda.find_eigen_values()
    for i in range(len(qda_eigen_vals)):
        print("QDA eigen values" ,qda_eigen_vals[i])

    lda_eigen_vals = lda.find_eigen_values()
    print("LDA eigen values", lda_eigen_vals)



if __name__ == '__main__':

    main()
