import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import mode

fileName = 'generated_train.csv'
dataAndID = np.genfromtxt(fileName, delimiter=',')

dataTrain = dataAndID[:, :-1]
labelTrain = dataAndID[:, -1]

fileName = 'generated_test.csv'
dataAndID = np.genfromtxt(fileName, delimiter=',')

dataTest = dataAndID[:, :-1]
labelTest = dataAndID[:, -1]


def getGaussConditionalPorbabilitybyParametric(X, mu, sigma):
    n = X.shape[0]
    d = X.shape[1]

    sigma_inv = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    z = np.zeros(n)

    for i in range(n):
        diff = X[i] - mu
        z[i] = (1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_sigma))) * \
               np.exp(-0.5 * diff @ sigma_inv @ diff)

    return z


def getGaussConditionalPorbabilitybyNonparametric(X, Y, h):
    z = []

    d = X.shape[1]
    k = Y.shape[0]

    for i in range(X.shape[0]):
        diff = X[i] - Y
        distances = np.sum(diff ** 2, axis=1)

        probability = np.sum(
            np.exp(-distances / (2 * h ** 2))
        ) / (k * (2 * np.pi * h ** 2) ** (d / 2))

        z.append(probability)

    return np.array(z)


plt.figure()
plt.scatter(dataTrain[:, 0], dataTrain[:, 1], s=5, c=labelTrain)
plt.title('Training dataset')

plt.figure()
plt.scatter(dataTest[:, 0], dataTest[:, 1], s=5, c=labelTest)
plt.title('Testing dataset')


uniqueID = np.unique(labelTrain)
numTrain = labelTrain.shape[0]


# Step 2: Parametric Bayesian classifier

y = []

for i in range(uniqueID.shape[0]):
    data = dataTrain[labelTrain == uniqueID[i], :]

    m = np.mean(data, axis=0)
    var = np.cov(data, rowvar=False)

    conditionalPDF = getGaussConditionalPorbabilitybyParametric(
        dataTest, m, var
    )

    prior = data.shape[0] / numTrain

    conditionalPDF = prior * conditionalPDF

    y.append(conditionalPDF)

value = np.max(y, axis=0)
predictedID = np.argmax(y, axis=0)

accuracy = np.sum(predictedID == labelTest) / labelTest.shape[0]

print("Parametric accuracy:", accuracy)

plt.figure()
plt.scatter(dataTest[:, 0], dataTest[:, 1], s=5, c=predictedID)
plt.title('Decision boundary of Bayesian classifier')
plt.show()


# Step 3: Naive Bayesian classifier

y = []

for i in range(uniqueID.shape[0]):
    data = dataTrain[labelTrain == uniqueID[i]]

    m = np.mean(data, axis=0)
    var = np.cov(data.T)

    var_naive = np.diag(np.diag(var))

    y.append(
        getGaussConditionalPorbabilitybyParametric(
            dataTest, m, var_naive
        )
    )

value, predicted_id = np.max(y, axis=0), np.argmax(y, axis=0)

accuracy = np.sum(predicted_id == labelTest) / labelTest.shape[0]

print("Naive Bayesian accuracy:", accuracy)

plt.figure()
plt.scatter(dataTest[:, 0], dataTest[:, 1], s=5, c=predicted_id)
plt.title('Decision boundary of Naive Bayesian classifier')
plt.show()


# Step 4: Non-parametric Bayesian classifier

h_values = [0.1, 0.3, 0.5, 1, 10]

for h in h_values:

    y = []

    for i in range(uniqueID.shape[0]):
        data_base = dataTrain[labelTrain == uniqueID[i]]

        conditional_pdf = getGaussConditionalPorbabilitybyNonparametric(
            dataTest,
            data_base,
            h
        )

        prior = data_base.shape[0] / numTrain

        conditional_pdf = prior * conditional_pdf

        y.append(conditional_pdf)

    value, predicted_id = np.max(y, axis=0), np.argmax(y, axis=0)

    accuracy = np.sum(predicted_id == labelTest) / labelTest.shape[0]

    print("Non-parametric h =", h, "accuracy:", accuracy)

    plt.figure()
    plt.scatter(
        dataTest[:, 0],
        dataTest[:, 1],
        s=5,
        c=predicted_id
    )

    plt.title(
        'Non-parametric Bayesian classifier h=' + str(h)
    )

    plt.show()


# Step 5: K-NN classifier

W = dataTest
P = dataTrain

Z = cdist(W, P, 'euclidean')

ind = np.argsort(Z, axis=1)

k_values = [1, 5, 15, 155]

for k in k_values:

    predictedKlabels = labelTrain[ind[:, :k]]

    predictedID = mode(
        predictedKlabels,
        axis=1
    ).mode.flatten()

    accuracy = np.sum(
        predictedID == labelTest
    ) / len(labelTest)

    print("K-NN k =", k, "accuracy:", accuracy)

    plt.figure()

    plt.scatter(
        dataTest[:, 0],
        dataTest[:, 1],
        s=5,
        c=predictedID
    )

    plt.title(
        'Decision boundary of KNN classifier k=' + str(k)
    )

    plt.show()
