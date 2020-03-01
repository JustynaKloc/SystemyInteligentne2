import numpy as np
import itertools
import operator

def calculate_entropy(x, y, r):
    x=np.array(x)
    y=np.array(y)
    all_results=[]
    entropy=[]
    for j in range(len(x)):
        point=x[j]
        result = []
        for i in range(len(x)):
            if i==j:
                continue;
            distance = np.linalg.norm(point-x[i])
            if distance<=r:
                result.append([distance,i,y[i]])      
        all_results.append(result)
        if result:
            entropy.append(1-((sum(np.array(result)[:,2]==y[j])) / len(result)))
        else:
            entropy.append(0)
    return all_results, entropy


def calculate_combinations(self):
    old_res = [ [i] for i in range(self.input_list[0].n_functions)]
    for i in range(1,self.input_number):  
        res=[]
        for j in range(self.input_list[i].n_functions):
            for k in range(len(old_res)):
                res.append([j] + old_res[k])
        old_res = res
    return res


def indeks_Jaccarda(mf, x):
    dx=x[1]-x[0]
    mins = []
    for a,b in itertools.combinations(mf.fuzzify(x).T, 2):
        mins.append(np.minimum(a,b))
    min_values = np.max(mins, axis=0)
    integral_min = sum((min_values[1:]+min_values[:-1])*0.5*dx)
    max_values = np.max(mf.fuzzify(x), axis=1)
    integral_max = sum((max_values[1:]+max_values[:-1])*0.5*dx)
    
    return integral_min/integral_max

def indeksPodzialuJednosci(mf, x):
    dx=x[1]-x[0]
    sum_val_mf = np.sum(mf.fuzzify(x), axis=1)
    counter = sum(np.abs(sum_val_mf-np.ones(np.shape(sum_val_mf))*0.5))
    denominator = (max(x) - min(x))/dx +1
    return counter/denominator

def findMembershipFunctions(data, labels):#c, w, fl, fr
    idx_sorted = np.argsort(data)
    mf = []
    i=0
    #last_end=data[idx_sorted[i]]
    s_labels = set(labels)
    for label in s_labels:
        #print(label)
        actual_input = data[labels==label]
        begin_f = min(actual_input)
        begin_f_idx = np.argwhere(data == begin_f)
        end_f = max(actual_input)
        #end_f_idx = np.argwhere(data == end_f)
        i = begin_f_idx

        r = max((list(y) for (x,y) in itertools.groupby((enumerate(labels[idx_sorted])),operator.itemgetter(1))
                 if x == label), key=len)
        
        begin = data[idx_sorted[r[0][0]]]
        end = data[idx_sorted[r[-1][0]]]
        mf.append([(end-begin)/2+begin, end-begin, begin-begin_f, end_f-end])
       
    return mf

def my_reshape(data, shaped_data):
    temp = []
    
    last = 0
    for i in range(len(shaped_data)):
        temp.append(data[last:last+len(shaped_data[i])])
        last = last+len(shaped_data[i])
    return temp
      