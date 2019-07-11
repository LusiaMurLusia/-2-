# Подключаем нужные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# создаем пустой array нужной длины
points = np.zeros((100, 2), float)
print(points[:5][:])

# заполняем его 4 группами точек (типа 4 кластера будет)
points[:25,0] = np.random.normal(1.5, 0.5,size=25)
points[:25,1] = np.random.normal(1.5, 0.5,size=25)

points[25:50,0] = np.random.normal(-1.5, 0.5,size=25)
points[25:50,1] = np.random.normal(1.5, 0.5,size=25)

points[50:75,0] = np.random.normal(-1.5, 0.5,size=25)
points[50:75,1] = np.random.normal(-1.5, 0.5,size=25)

points[75:101,0] = np.random.normal(1.5, 0.5,size=25)
points[75:101,1] = np.random.normal(-1.5, 0.5,size=25)

print(points[:5,:])

# смотрим, как создались
plt.plot(points[:,0],points[:,1],'ro')
plt.rcParams['figure.figsize']=(10,5)
plt.show(10)

# создаем случайные 4 точки, равномерно распределённые по всему используемому пространству
clasters = np.random.uniform(-3,3,8).reshape((4,2))
print(clasters)

# совмещаем 2 графика и смотрим
plt.subplot(1, 1, 1)
plt.plot(clasters[:,0],clasters[:,1], 'kx',2)
plt.plot(points[:,0],points[:,1],'ro')
plt.show(10)

# будем хранить местоположения центроидов
cl_history = []
cl_history.append(clasters)

# подключаем библиотеку для удобного подсчёта расстояний между точками
from scipy.spatial.distance import cdist

# вот тут и происходит сам алгоритм k-means
for i in range(7):
    # Считаем расстояния от наблюдений до центроид
    distances = cdist(points, clasters)
    # Смотрим, до какой центроиде каждой точке ближе всего
    labels = distances.argmin(axis=1)

    # Положим в каждую новую центроиду геометрический центр её точек
    clasters = clasters.copy()
    clasters[0, :] = np.mean(points[labels == 0, :], axis=0)
    clasters[1, :] = np.mean(points[labels == 1, :], axis=0)
    clasters[2, :] = np.mean(points[labels == 2, :], axis=0)
    clasters[3, :] = np.mean(points[labels == 3, :], axis=0)

    cl_history.append(clasters)

# А теперь нарисуем всё это
plt.figure(figsize=(15, 10))
for i in range(6):
    distances = cdist(points, cl_history[i])
    labels = distances.argmin(axis=1)

    plt.subplot(3, 2, i + 1)
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'ro', label='cluster #1')
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'go', label='cluster #2')
    plt.plot(points[labels == 2, 0], points[labels == 2, 1], 'bo', label='cluster #3')
    plt.plot(points[labels == 3, 0], points[labels == 3, 1], 'yo', label='cluster #4')
    plt.plot(cl_history[i][:, 0], cl_history[i][:, 1], 'kX')
    plt.legend(loc=0)
    plt.title('Step {:}'.format(i + 1));
plt.show(10)





