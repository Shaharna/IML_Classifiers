import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy.special as scs

def question_5():
    """
    The answer to question number five
    :return:
    """
    x = np.arange(-10, 10, 0.001)

    # j:
    fig = plt.figure()
    # creating a normal distribution with mean 4 and scale 1
    cdf0 = scipy.stats.norm.cdf(x, 4, 1)

    # creating a normal distribution with mean 6 and scale 1
    cdf1 = scipy.stats.norm.cdf(x, 6, 1)

    plt.title("CDF")
    plt.xlabel("x")
    plt.ylabel("Probability density")

    plt.plot(x, cdf0, label = "Mean = 4", color="orange")
    plt.plot(x, cdf1, label = "Mean = 6", color="purple")

    plt.legend()
    plt.show()

    # creating a normal distribution with mean 4 and scale 1
    pdf0 = scipy.stats.norm.pdf(x, 4, 1)

    # creating a normal distribution with mean 6 and scale 1
    pdf1 = scipy.stats.norm.pdf(x, 6, 1)

    plt.title("PDF")
    plt.xlabel("x")
    plt.ylabel("Probability density")

    plt.plot(x, pdf0, label="Mean = 4", color="orange")
    plt.plot(x, pdf1, label="Mean = 6", color="purple")

    plt.legend()
    plt.show()

    # ii:
    mu_0 = 4
    mu_1 = 6
    pai_0 = 0.5
    pai_1 = 0.5
    w0 = -pai_1 *(mu_1*(1)*mu_1) + pai_0 *(mu_0*(1)*mu_0)
    w = 2

    h = scs.expit(w*x+ w0)
    plt.title("h(x) as a function of x")
    plt.xlabel("x")
    plt.ylabel("h(x)")

    plt.plot(x, h, color="purple")

    plt.show()

    # iii:
    x1= np.arange(0, 1, 0.001)
    h1 = (scs.logit(x1)-w0)/ w
    cdf_h0 = scipy.stats.norm.cdf(h1, 4, 1)

    plt.title("CDF of h(x) for x~X| Y = 0")
    plt.xlabel("x")
    plt.ylabel("Probability density")

    plt.plot(x1, cdf_h0, label="Mean = 4", color="orange")

    plt.legend()
    plt.show()

    cdf_h1 = scipy.stats.norm.cdf(h1, 6, 1)

    plt.title("CDF of h(x) for x~X| Y = 1")
    plt.xlabel("x")
    plt.ylabel("Probability density")

    plt.plot(x1, cdf_h1, label="Mean = 6", color="purple")

    plt.legend()
    plt.show()

    # iv:

    x2 = np.arange(0, 1.00, 0.001)

    z1 = cdf_h0
    z2 = cdf_h1

    plt.title("1- CDF of h(x) for z1~X| Y = 0, z2~X| Y = 1")
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.plot(x2, 1-z1, label="z1~X| Y = 0", color="orange")
    plt.plot(x2, 1 - z2, label="z2~X| Y = 1", color="purple")

    plt.legend()
    plt.show()


    # vi:

    x3= np.arange(-10, 10, 0.001)

    # gaussian
    pdf_0 = scipy.stats.norm.pdf(x, 4, 1)
    pdf_1 = scipy.stats.norm.pdf(x, 6, 1)
    plt.title("The PDF with Thresholds of h(x)")
    plt.ylabel('density')
    plt.xlabel('X')
    plt.plot(x3, pdf_0, label="Mu 0= 4")
    plt.plot(x3, pdf_1, label="Mu 1= 6")

    plt.axvline(x=(scipy.special.logit(0.2) - w0) / w,
                color="red", label="0.2 ", linestyle="--")
    plt.axvline(x=(scipy.special.logit(0.4) - w0) / w,
                color="blue", label="0.4 ", linestyle="--")
    plt.axvline(x=(scipy.special.logit(0.55) - w0) / w,
                color="green", label="0.55 ", linestyle="--")
    plt.axvline(x=(scipy.special.logit(0.95) - w0) / w,
                color="orange", label="0.95 ", linestyle="--")

    plt.legend()
    plt.show()

    #vii:

    plt.title("RCO curve of h")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(1-z1, 1 - z2, color="orange")

    plt.show()


if __name__ == '__main__':
    question_5()
