from random import randrange
import random

import numpy as np
from sklearn.model_selection import train_test_split

from ANFIS import ANFIS
from params import FuzzyInputVariable_2Sigmoids


def generate_initial_population_for_anfis(population_number: int):
    population = [[round(random.uniform(0.0, 1.0), 5) for i in range(46)] for j in range(population_number)]
    return population


def anfis_fitness_function(x: list):
    var_list, data, labels = generate_anfis_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=25)

    fis = ANFIS(var_list, X_train.T, y_train)
    return fis.get_absolute_error(x)


def generate_anfis_data():
    x = np.arange(0.1, 10, 0.5)
    x, y, z = np.meshgrid(x, x, x)

    dataX = x.flatten()
    dataY = y.flatten()
    dataZ = z.flatten()
    dataXYZ = np.column_stack((dataX, dataY, dataZ))

    output = (dataX ** 0.5 + dataY ** (-1) + dataZ ** 1.5) ** 2

    varX = FuzzyInputVariable_2Sigmoids(0.5, 0.5, "XAxis", ["L", "H"])
    varY = FuzzyInputVariable_2Sigmoids(0.5, 0.5, "XAxis", ["L", "H"])
    varZ = FuzzyInputVariable_2Sigmoids(0.5, 0.5, "XAxis", ["L", "H"])

    varX.show()

    return [varX, varY, varZ], dataXYZ, output
