import numpy as np
from sklearn.model_selection import train_test_split

from genetic_util import generate_initial_population_for_anfis
import gen_algorythm
from ANFIS import ANFIS


x, y = gen_algorythm.run_gen_algorithm(generate_initial_population_for_anfis(100), 10)

population = x[-1]
p = population['Individuals']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(p[0], p[0], p[0])
#plt.show()
