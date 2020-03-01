# Algorytm roju cząstek

# Wybiermay najlepszą cząsteczke z roju
# Każdą pozycje zapamiętujemy jako najlepszą pozycję tej cząsteczki
# Nadajemy im prędkości i kierunek
# Iterujemy przez wszystkie cząsteczki i modyfikujemy prędkość biorąc pod uwagę trzy czynniki
# V = a1 * v + a2 * c + a3 * r
# c - mądrość samej cząsteczki
# r - mądrość roju
# a3 * r(R-x)
# r - liczba losowa [0; 1]
# x = x + V
# Nadpisujemy cząsteczkę jeśli osiągnęła lepszą pozycję i wartość najlepszej cząsteczki

# V = p1 * V + c1 * r1 (C - x) + c2 * r2 * (R - x)
# c1,c2 - jak czasteczka ufa samej sobie
# r1 - liczba losowa
# p3 - jak ufam calemu rojowi

# 0.5,

import random

import numpy as np
import genetic_util


def fitness_function(position):
	fitness = round(genetic_util.anfis_fitness_function(position), 4)
	return fitness


particle_position_vector = genetic_util.generate_initial_population_for_anfis(100)

W = 0.5  # szybkość
c1 = 0.5  # ufanie sobie
c2 = 0.9  # ufanie rojowi
target = 0.04

n_iterations = 50
target_error = 1
n_particles = 30


fitness_value = [round(genetic_util.anfis_fitness_function(item)) for item in particle_position_vector]
sorted_fitness = sorted([[particle_position_vector[x], fitness_value[x]] for x in range(len(particle_position_vector))],
						key=lambda x: x[1])

pbest_fitness_value = [sorted_fitness[x][1] for x in range(len(sorted_fitness))]
pbest_position = [sorted_fitness[x][0] for x in range(len(sorted_fitness))]

gbest_fitness_value = pbest_fitness_value[0]
gbest_position = pbest_position[0]

velocity_vector = [[round(random.uniform(0.0, 1.0), 5) for p in pbest_position[0]] for pos in pbest_position]
print(velocity_vector)

n_particles = len(pbest_position)
n_moves = 10
iteration = 0

def new_velocity_funk(W, c1, c2, rand1, rand2, velocity_vector, pbest_position, particle_position_vector, gbest_position):
	velocity = list()
	for j in range(len(velocity_vector)):
		velocity.append((W * velocity_vector[j]) + (c1 * rand1) * (pbest_position[j] - particle_position_vector[j]) + (
				c2 * rand2) * (gbest_position[j] - particle_position_vector[j]))
	return velocity


for m in range(n_moves):
	for i in range(n_particles):
		fitness_cadidate = fitness_function(particle_position_vector[i])
		
		if pbest_fitness_value[i] > fitness_cadidate:
			pbest_fitness_value[i] = fitness_cadidate
			pbest_position[i] = particle_position_vector[i]
		
		if gbest_fitness_value > fitness_cadidate:
			gbest_fitness_value = fitness_cadidate
			gbest_position = particle_position_vector[i]
	
	if abs(gbest_fitness_value - target) < target_error:
		break
	
	rand1 = round(random.uniform(0.0, 1.0), 5)
	rand2 = round(random.uniform(0.0, 1.0), 5)
	for i in range(n_particles):
		new_velocity = new_velocity_funk(W, c1, c2, rand1, rand2, velocity_vector[i], pbest_position[i], particle_position_vector[i], gbest_position)
		new_position = list()
		for j in range(len(new_velocity)):
			new_position.append(new_velocity[j] + particle_position_vector[i][j])
		particle_position_vector[i] = new_position
	print("numer iteracji ", iteration +1, "błąd", gbest_fitness_value) # "najlepsza pozycja ", gbest_position )


	iteration = iteration + 1

print( "numer iteracji ", iteration,"błąd:",gbest_fitness_value, "\n najlepsza pozycja", gbest_position,)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
fig = plt.figure()

#dataXYZ = np.arange(0.1, len(gbest_position))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gbest_position, gbest_position)
plt.show()
