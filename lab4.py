from params import FuzzyInputVariable_3Trapezoids, FuzzyInputVariable_2Trapezoids, FuzzyInputVariable_List_Trapezoids
from operators import productN, zadehN, lukasiewiczN, drasticN, einsteinN, fodorN, tnorm
import numpy as np
import matplotlib.pyplot as plt
from ANFIS import ANFIS
import time
import copy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

x = np.arange(1, 12, 1)
x,y,z = np.meshgrid(x, x, x)

dataX = x.flatten()
dataY = y.flatten()
dataZ = z.flatten()
dataXYZ = np.column_stack((dataX,dataY,dataZ))

data_labels = []
for x, y, z in zip(dataX, dataY, dataZ):
    data_labels.append(((x ** (0.5) + 1/ y + z**(1.5))) ** 2) 
data_labels=np.array(data_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(dataX, dataY,dataZ ,data_labels, c=data_labels)


varX = FuzzyInputVariable_2Trapezoids(5, 0.5, "XAxis", ["L","M"])
varY = FuzzyInputVariable_2Trapezoids(5, 0.5, "YAxis", ["L","M"])
varZ = FuzzyInputVariable_2Trapezoids(5, 0.5, "ZAxis", ["L","M"])

#Wyświetlanie funkcji przynależnosci
plt.figure()
varX.show(x = np.arange(0, 10, 0.01))
plt.legend()

plt.figure()
varY.show(x = np.arange(0, 10, 0.01))
plt.legend()

plt.figure()
varZ.show(x = np.arange(0, 10, 0.01))
plt.legend()

X_train, X_test, y_train, y_test = train_test_split(dataXYZ, data_labels, test_size=0.2, random_state=25)

fis = ANFIS([varX, varY, varZ], X_train.T, y_train)

start = time.time()
fis.train(True, True, False, True, n_iter=300) #true- optymalizacja globalna,czy optymalizujemy przesłanki, czy operator, czy konkluzej
end = time.time()
print("FIS premises", fis.premises)
print("TIME elapsed: ", end - start)   
fis.training_data = X_train.T
fis.expected_labels = y_train
fis.show_results(y_train)