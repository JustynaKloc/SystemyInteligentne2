# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.stats import variation
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from helps_and_enhancers import calculate_combinations, my_reshape
from operators import productN
from params import FuzzyInputVariable_3Trapezoids, FuzzyInputVariable_2Trapezoids

from goal_function_object import *


class ANFIS:

    def __init__(self, inputs, training_data: np.ndarray, expected_labels: np.ndarray, operator_function=productN,
                 operator_init_value=0.5):
        self.input_list = inputs
        self.input_number = len(inputs)
        self.training_data = training_data
        self.expected_labels = expected_labels

        self.premises = []
        for i in range(self.input_number):
            self.premises.append(self.input_list[i].get())

        self.nodes_number = np.prod([inp.n_functions for inp in self.input_list])

        self.operator_function = operator_function

        # self.tsk = np.ones((self.nodes_number ,self.input_number+1))
        self.tsk = np.random.random((self.nodes_number, self.input_number + 1))

        self.op = [operator_init_value] * self.nodes_number

        self.calculate_aids()

    # Wyswietlanie funkcji przynależnosci
    def show_inputs(self):
        plt.figure()
        for i in range(self.input_number):
            plt.subplot(self.input_number, 1, i + 1)
            self.input_list[i].show()
            plt.legend()
        plt.show()

    def set_premises_parameters(self, fv):
        fv = np.array(fv).reshape(np.shape(self.premises))
        self.premises = fv
        for i in range(self.input_number):
            self.input_list[i].set(*fv[i])

    def calculate_aids(self):
        self.premises_combinations = np.array(calculate_combinations(self))[:, ::-1]
        x1 = [item for sublist in self.premises for item in sublist]
        x1 = np.array(x1).flatten()
        x2 = self.op
        self.end_x1 = len(x1)
        self.end_x2 = len(x1) + len(x2)

    def output_to_labels(self, y_pred):
        rounded = np.round(y_pred.flatten()).astype(int)
        r_shape = np.shape(rounded)
        return np.max((np.min((rounded, np.ones(r_shape)), axis=0), np.zeros(r_shape)), axis=0)  # clamp 0-1

    def anfis_estimate_labels(self, fv, op, tsk) -> np.ndarray:

        data = self.training_data

        self.set_premises_parameters(fv)
        tsk = np.reshape(tsk, np.shape(self.tsk))
        memberships = [self.input_list[x].fuzzify(data[x]) for x in range(self.input_number)]

        # Wnioskowanie
        arguments = []
        for premises in self.premises_combinations:
            item = []
            for i in range(len(premises)):
                item.append(np.array(memberships[i])[:, premises[i]])
            arguments.append(item)

        arguments = np.transpose(arguments, (1, 2, 0))

        R = self.operator_function(arguments, op)

        # Normalizacja normalizacja poziomów aktywacji reguł
        Rsum = np.sum(R, axis=1, keepdims=True)

        Rnorm = R / Rsum
        Rnorm[(Rsum == 0).flatten(), :] = 0
        # wylicz wartoci przesłanek dla każdej próbki

        dataXYZ1 = np.vstack((self.training_data, np.ones(len(self.training_data[0])))).T
        Q = np.dot(dataXYZ1, tsk.T)

        # wyznacz wyniki wnioskowania dla każdej próbki
        result = (Q * Rnorm).sum(axis=1, keepdims=True)

        return result.T

    def show_results(self, color=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if color is None:
            color = [[1, 0, 0] if cc else [0, 1, 0] for cc in self.expected_labels]

        result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)

        # ax.scatter(np.array(self.training_data)[:,0], np.array(self.training_data)[:,1], result, c=rgb)
        ax.scatter(self.training_data[0], self.training_data[1], result, c=color)

        plt.show()

    def show_3d_results(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)

        ax.scatter(self.training_data[0], self.training_data[1], self.training_data[2], c=result.flatten(),
                   cmap=plt.inferno())

        fig.canvas.set_window_title('Results')
        plt.show()

    def show_3d_results_for_x(self, x:list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        input = np.array(x)
        fv = input[:self.end_x1].reshape(np.shape(self.premises))
        op = input[self.end_x1:self.end_x2]
        tsk = input[self.end_x2:]
        result = self.anfis_estimate_labels(fv, op, tsk)

        ax.scatter(self.training_data[0], self.training_data[1], self.training_data[2], c=result.flatten(),
                   cmap=plt.inferno())

        fig.canvas.set_window_title('Results')
        plt.show()

    def get_absolute_error(self, input: list):
        # result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)
        # return np.abs(expected_results - result)
        input = np.array(input)
        # print('input', input)
        fv = input[:self.end_x1].reshape(np.shape(self.premises))
        # print(fv)
        op = input[self.end_x1:self.end_x2]
        tsk = input[self.end_x2:]
        new_labels = self.anfis_estimate_labels(fv, op, tsk)

        error = (np.abs(new_labels - self.expected_labels)).sum()
        return error

    def show_abs_error_results(self, expected_results):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)
        error_result = np.abs(expected_results - result)

        img = ax.scatter(self.training_data[0], self.training_data[1], self.training_data[2], c=error_result.flatten(),
                         cmap=plt.cool())
        fig.colorbar(img)
        fig.canvas.set_window_title('Absolute error')
        plt.show()

    def show_relative_error_results(self, expected_results):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)
        error_result = np.abs(expected_results - result) / expected_results

        img = ax.scatter(self.training_data[0], self.training_data[1], self.training_data[2], c=error_result.flatten(),
                         cmap=plt.cool())
        fig.colorbar(img)
        fig.canvas.set_window_title('Relative error')
        plt.show()

    def set_training_and_testing_data(self, training_data, expected_labels):
        self.training_data = training_data
        self.expected_labels = expected_labels

    def get_merged_anfis_parameters(self):
        x1 = [item for sublist in self.premises for item in sublist]
        x1 = np.array(x1).flatten()
        x2 = self.op
        x3 = self.tsk.flatten()
        x0 = np.hstack((x1, x2, x3))
        return x0

    def train(self, global_optimization: bool, learn_premises: bool, learn_operators: bool, learn_consequents: bool,
              n_iter=100, bounds_premises=None):
        global_optimization = False

        x1 = [item for sublist in self.premises for item in sublist]
        x1 = np.array(x1).flatten()
        x2 = self.op
        x3 = self.tsk.flatten()

        if bounds_premises is None:
            bfv = [(0, 4)] * len(x1)
        else:
            bfv = bounds_premises
        bop = [(0.0, 2.0)] * len(x2)
        btsk = [(0, 2)] * len(x3)

        niter_success = 100

        if learn_premises and learn_operators and learn_consequents:
            x0 = np.hstack((x1, x2, x3))
            self.end_x1 = len(x1)
            self.end_x2 = len(x1) + len(x2)

            bounds = bfv + bop + btsk

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds, "args": (self)}
                res = basinhopping(goal_premises_operators_consequents, x0, minimizer_kwargs=minimizer_kwargs,
                                   niter=n_iter, niter_success=niter_success)
            else:
                res = minimize(goal_premises_operators_consequents, x0, method='SLSQP', bounds=bounds, args=self)

            self.set_premises_parameters(res.x[:self.end_x1].reshape(np.shape(self.premises)))
            self.op = res.x[self.end_x1:self.end_x2]
            self.tsk = res.x[self.end_x2:].reshape(np.shape(self.tsk))

        elif learn_premises and learn_operators:
            x0 = np.hstack((x1, x2))
            self.end_x1 = len(x1)
            self.end_x2 = len(x1) + len(x2)

            bounds = bfv + bop

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_premises_operators, x0, minimizer_kwargs=minimizer_kwargs, niter=n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_premises_operators, x0, method='SLSQP', bounds=bounds, args=self)

            self.set_premises_parameters(res.x[:self.end_x1].reshape(np.shape(self.premises)))
            self.op = res.x[self.end_x1:self.end_x2]

        elif learn_premises and learn_consequents:
            x0 = np.hstack((x1, x3))
            self.end_x1 = len(x1)
            self.end_x2 = len(x1)

            bounds = bfv + btsk

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds, "args": (self)}
                res = basinhopping(goal_premises_consequents, x0, minimizer_kwargs=minimizer_kwargs,
                                   niter=n_iter)  # , niter_success=niter_success)
            else:
                res = minimize(goal_premises_consequents, x0, method='SLSQP', bounds=bounds, args=self, tol=1e-6)

            self.set_premises_parameters(res.x[:self.end_x1])  ##zmiana funkcji
            self.tsk = res.x[self.end_x2:].reshape(np.shape(self.tsk))

        elif learn_operators and learn_consequents:
            print("4")
            x0 = np.hstack((x2, x3))
            self.end_x1 = 0
            self.end_x2 = len(x2)

            bounds = bop + btsk

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_operators_consequents, x0, minimizer_kwargs=minimizer_kwargs, niter=n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_operators_consequents, x0, method='SLSQP', bounds=bounds, args=self)

            self.op = res.x[self.end_x1:self.end_x2]
            self.tsk = res.x[self.end_x2:].reshape(np.shape(self.tsk))

        elif learn_premises:
            x0 = x1
            self.end_x1 = len(x1)
            self.end_x2 = len(x1)

            bounds = bfv

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_premises, x0, minimizer_kwargs=minimizer_kwargs, niter=n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_premises, x0, method='SLSQP', bounds=bounds, args=self)

            self.set_premises_parameters(res.x[:].reshape(np.shape(self.premises)))

        elif learn_operators:
            x0 = x2
            self.end_x1 = 0
            self.end_x2 = len(x2)

            bounds = bop

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
                res = basinhopping(goal_operators, x0, minimizer_kwargs=minimizer_kwargs, niter=n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_operators, x0, method='SLSQP', bounds=bounds, args=self)

            self.op = res.x[:]

        elif learn_consequents:
            x0 = x3
            self.end_x1 = 0
            self.end_x2 = 0

            bounds = btsk

            if global_optimization:
                minimizer_kwargs = {"method": "SLSQP", "bounds": bounds, "args": (self), "tol": 1e-03}
                res = basinhopping(goal_consequents, x0, minimizer_kwargs=minimizer_kwargs, niter=n_iter,
                                   niter_success=niter_success)
            else:
                res = minimize(goal_consequents, x0, method='SLSQP', bounds=bounds, args=self)

            self.tsk = res.x[:].reshape(np.shape(self.tsk))

        else:
            print("Error")
            assert (0)

        print("Optymalizacja zakończona!")
        print("z blędem:  ", res.fun)
        print("Liczba ew: ", res.nfev)
        print("Liczba it: ", res.nit)

        return res.fun

