from scipy.optimize import fmin_tnc
import numpy as np 
import math

# x[0] = R1
# x[1] = L1
# x[2] = L2
# x[3] = C1
# x[4] = C2
# x[5] = R2

def f(x):
    k = 0.1
    o = 2*math.pi*50
    return np.arctan(((x[1]*x[2]*x[0]*k^2*o^2*(x[2]*o-1/(x[4]*o))+x[1]*x[2]*x[5]*k^2*o^2*(x[1]*o-1/(x[3]*o)))/((x[2]*o-1/(x[4]*o))^2*((x[1]*o-1/x[3]*o))^2+x[0]^2)+x[5]^2*(x[1]*o-1/x[3]*o))^2+x[1]*x[2]*x[0]*x[5]*k^2*o^2-x[1]*x[2]*k^2*o^2*(x[1]*o-1/(x[3]*o))*(x[2]*o-1/(x[4]*o))+x[0]^2*x[5]^2)

g = np.array([10,0.1,0.1,0.0001,0.000001,100])
b = [(0.001,1000000),(0.000001,20),(0.000000000001,0.000001)]
fmin_tnc(f,g,bounds=b)
