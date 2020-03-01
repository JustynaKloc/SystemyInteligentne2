import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss
from genetic_util import anfis_fitness_function


def individual(number_of_genes, upper_limit, lower_limit):
    """ Funkcja zbiera ilość genów(number_of_genes)
        max i min wartości dla genów(upper_limit, lower_limit)
        i zwraca jednostkę w postaci listy"""
    individual = [round(rnd() * (upper_limit - lower_limit)
                        + lower_limit, 1) for x in range(number_of_genes)]
    # randomowa generacja genów dla jednostki populacji
    return individual


def population(number_of_individuals,
               number_of_genes, upper_limit, lower_limit):
    """Funkcja tworząca populacje o zadanej liczbie
       jednostek(number_of_individuals) o zadanej liczbie
       genów(number_of_genes)
       zwraca listę"""
    return [individual(number_of_genes, upper_limit, lower_limit)
            for x in range(number_of_individuals)]


def fitness_calculation(individual):
    """obecne: Sum of all genes"""
    fitness_value = round(anfis_fitness_function(individual), 4)
    return fitness_value


# 3 rodzaję liniowa, kwadratowa i logarytmiczna
def roulette(cum_sum, chance):
    """Roulet selection"""
    veriable = list(cum_sum.copy())
    veriable.append(chance)
    veriable = sorted(veriable)
    return veriable.index(chance)



def mutation(individual, upper_limit, lower_limit, muatation_rate=2,
             method='Gauss', standard_deviation=0.001):
    gene = [randint(0, 7)]
    for x in range(muatation_rate - 1):
        gene.append(randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = randint(0, 7)
    mutated_individual = individual.copy()

    if method == 'Reset':
        for x in range(muatation_rate):
            mutated_individual[x] = round(rnd() * \
                                          (upper_limit - lower_limit) + lower_limit, 1)
    return mutated_individual




def selection(generation, method='Fittest Half'):
    generation['Normalized Fitness'] = \
        sorted([generation['Fitness'][x] / sum(generation['Fitness'])
                for x in range(len(generation['Fitness']))], reverse=True)
    generation['Cumulative Sum'] = np.array(
        generation['Normalized Fitness']).cumsum()
    if method == 'Roulette Wheel':
        selected = []
        for x in range(len(generation['Individuals']) // 2):
            selected.append(roulette(generation
                                     ['Cumulative Sum'], rnd()))
            while len(set(selected)) != len(selected):
                selected[x] = \
                    (roulette(generation['Cumulative Sum'], rnd()))
        selected = {'Individuals':
                        [generation['Individuals'][int(selected[x])]
                         for x in range(len(generation['Individuals']) // 2)]
            , 'Fitness': [generation['Fitness'][int(selected[x])]
                          for x in range(
                    len(generation['Individuals']) // 2)]}
    elif method == 'Fittest Half':
        # selected_individuals = [generation['Individuals'][-x - 1]
        #                         for x in range(int(len(generation['Individuals']) // 2))]
        # selected_fitnesses = [generation['Fitness'][-x - 1]
        #                       for x in range(int(len(generation['Individuals']) // 2))]
        selected_individuals = [generation['Individuals'][x]
                                for x in range(int(len(generation['Individuals']) // 2))]
        selected_fitnesses = [generation['Fitness'][x]
                              for x in range(int(len(generation['Individuals']) // 2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}
    elif method == 'Random':
        selected_individuals = \
            [generation['Individuals']
             [randint(1, len(generation['Fitness']))]
             for x in range(int(len(generation['Individuals']) // 2))]
        selected_fitnesses = [generation['Fitness'][-x - 1]
                              for x in range(int(len(generation['Individuals']) // 2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}
    return selected


def next_generation(gen, upper_limit, lower_limit):
    """Wyszukiwanie następnej generacji"""
    elit = {}
    next_gen = {}
    elit['Individuals'] = gen['Individuals'].pop(0)
    elit['Fitness'] = gen['Fitness'].pop(0)
    selected = selection(gen)
    parents = pairing(elit, selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]
    unmutated = selected['Individuals'] + offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit)
               for x in range(len(gen['Individuals']))]
    unsorted_individuals = mutated + [elit['Individuals']]
    unsorted_next_gen = \
        [fitness_calculation(mutated[x])
         for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x]
                        for x in range(len(gen['Fitness']))] + [elit['Fitness']]
    sorted_next_gen = \
        sorted([[unsorted_individuals[x], unsorted_fitness[x]]
                for x in range(len(unsorted_individuals))],
               key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0]
                               for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1]
                           for x in range(len(sorted_next_gen))]
    gen['Individuals'].append(elit['Individuals'])
    gen['Fitness'].append(elit['Fitness'])
    return next_gen


def pairing(elit, selected, method='Fittest'):
    global parents
    individuals = [elit['Individuals']] + selected['Individuals']
    fitness = [elit['Fitness']] + selected['Fitness']
    if method == 'Fittest':
        parents = [[individuals[x], individuals[x + 1]]
                   for x in range(len(individuals) // 2)]

    return parents



def mating(parents, method='Single Point'):
    global offsprings
    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0] \
                          [0:pivot_point] + parents[1][pivot_point:], parents[1]
                      [0:pivot_point] + parents[0][pivot_point:]]
    return offsprings



def first_generation(pop):
    fitness = [fitness_calculation(pop[x])
               for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
                             for x in range(len(pop))], key=lambda x: x[1])
    # print('sorted_fitness', sorted_fitness)
    population = [sorted_fitness[x][0]
                  for x in range(len(sorted_fitness))]
    # print('population', population)
    fitness = [sorted_fitness[x][1]
               for x in range(len(sorted_fitness))]
    # print(fitness)
    return {'Individuals': population, 'Fitness': sorted(fitness)}


def run_gen_algorithm(populations: list, num_of_generations: int):
    pop = populations
    #print("input population\n", pop)
    gen = list()
    gen.append(first_generation(pop))

    fitness_avg = [round(anfis_fitness_function(item), 4) for item in gen[0]['Individuals']]
    #print("first fitness\n błąd", fitness_avg)
    # fitness_max = np.array([max(gen[0]['Fitness'])])
    fitness_max = max(fitness_avg)
    print("numer iteracji: 1  najmniejszy błąd", fitness_max)
    first_last_fitness = list()
    first_last_fitness.append(fitness_max)

    for i in range(0, num_of_generations):
        gen.append(next_generation(gen[-1], 1, 0))
        #print('new_generation', next_generation(gen[-1], 1, 0))
        fitness_avg = [round(anfis_fitness_function(item), 4) for item in gen[-1]['Individuals']]
        #print("next fitness\nbłąd", fitness_avg)
        fitness_max = max(fitness_avg)
        print("numer iteracji:",i+2," najmniejszy błąd", fitness_max)

    first_last_fitness.append(fitness_max)
    print('początkowy i końcowy błąd',first_last_fitness)
    return gen, fitness_max


