from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from numpy import loadtxt, mean

path = "./database/wine_quality/winequality-red.csv"
data = loadtxt(path, skiprows=1, delimiter=',')

Y = data[:,11]

Y = Y.reshape(-1,1)

b = Binarizer(threshold=5)

Y = b.fit_transform(Y)

Y = Y.ravel()

X = data[:,:11]

scaler = StandardScaler()

X = scaler.fit_transform(X)

num_folds = 3

model = MLPClassifier(solver="lbfgs", activation="logistic", hidden_layer_sizes=(50,),  max_iter=2000, random_state=0)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

scores = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')

print(scores)
print(mean(scores))