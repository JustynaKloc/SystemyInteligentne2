import numpy as np

def productN(args, op): #działa
    return np.product(args, axis=0)

#nowa z operatorem
def tnorm(args, op):
    ones = np.ones(len(op))
    min_val = np.minimum(args[0],args[1])
    max_val = np.maximum(args[0],args[1])
    return np.multiply(op,min_val) + np.multiply((ones-op),max_val)

def zadehN(args, op): #działa
    return np.minimum(args[0],args[1])

def lukasiewiczN(args,op): #działa
    zero = np.zeros((len(args[0]),np.size(args[0],1)))
    ones = np.ones((len(args[0]),np.size(args[0],1)))
    return np.maximum(zero,np.subtract(np.add(args[0],args[1]),ones))
                   
def einsteinN(args,op):
    twos = np.full((len(args[0]), np.size(args[0], axis = 1)), 2)
    return np.divide(np.product(args, axis=0),np.subtract(twos,np.subtract(np.add(args[0], args[1]), np.product(args, axis=0))))

def fodorN(args, op):
    return pomocnicza(args, fodor_t)

def drasticN(args, op):
    return pomocnicza(args, drastic_t)

def pomocnicza(args, t_norm):
    result = np.zeros(args[0].shape, dtype=float)
    for i in range(0, len(args[0])):
        for j in range(0, len(args[0][0])):
            result[i][j] = t_norm(args[0][i][j], args[1][i][j])
    return result

def fodor_t(x, y):
    if x + y > 1:
        return min(x, y)
    else:
        return 0


def drastic_t(x, y):
    if x == 1:
        return y
    elif y == 1:
        return x
    else:
        return 0
'''
(2,700,4)
2 -> [x,y]
700 - liczba danych treningowych
4- liczba reguł 
'''