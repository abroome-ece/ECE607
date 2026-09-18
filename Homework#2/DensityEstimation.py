
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

file_name = 'Gaussian.csv'
data_with_id = np.genfromtxt(file_name, delimiter=',')
data = data_with_id[:, :-1]
ID = data_with_id[:, -1]
unique_id = np.unique(ID)

plt.scatter(data[ID == 0, 0], data[ID == 0, 1], color='red', label='Class 0')
plt.scatter(data[ID == 1, 0], data[ID == 1, 1], color='blue', label='Class 1')
plt.legend()
plt.show()

x_min = np.min(data[:, 0])
x_max = np.max(data[:, 0])
y_min = np.min(data[:, 1])
y_max = np.max(data[:, 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Parametric method')

for i in range(len(unique_id)):
    m = np.mean(data[ID == unique_id[i]], axis=0)
    var = np.cov(data[ID == unique_id[i]].T)
    f1 = multivariate_normal(mean=m, cov=var)
    x, y = np.mgrid[x_min:x_max:.1, y_min:y_max:.1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    ax.plot_surface(x, y, f1.pdf(pos), cmap='viridis')

plt.show()

h_values = [0.1, 0.3, 0.5, 1, 10]

x, y = np.mgrid[x_min:x_max:.1, y_min:y_max:.1]

for h in h_values:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Non-parametric method h=' + str(h))

    for i in range(len(unique_id)):
        tmp_data = data[ID == unique_id[i]]
        z = np.zeros(x.shape)

        for j in range(tmp_data.shape[0]):
            z += 1/(2*np.pi*h**2)*np.exp(
                -1/(2*h**2) *
                ((x-tmp_data[j, 0])**2 + (y-tmp_data[j, 1])**2)
            )

        z = z/tmp_data.shape[0]
        ax.plot_surface(x, y, z, cmap='viridis')

    plt.show()