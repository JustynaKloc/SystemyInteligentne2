# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:08:45 2019

@author: iperenc
"""
import numpy as np

def goal_premises_operators_consequents(input, self):
            
    fv = input[:self.end_x1].reshape(np.shape(self.premises))
    op  = input[self.end_x1:self.end_x2]
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    #error = (np.abs(new_labels - self.expected_labels)*self.entropy ).sum()
    #error = np.sqrt(np.abs(new_labels - dataC).sum())
    return error
    
def goal_premises_operators(input, self):
            
    fv = input[:self.end_x1].reshape(np.shape(self.premises))
    op  = input[self.end_x1:self.end_x2]
    tsk = self.tsk
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    return error

def goal_premises_consequents(input, self):

    fv = []
    last=0
    for i in range(len(self.premises)):
        fv.append(input[last:last + len(self.premises[i])])
        last = len(fv)
    fv = np.reshape(input[:self.end_x2], np.shape(self.premises))  # np.array(fv)
    op = self.op
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    return error

def goal_operators_consequents(input, self):
            
    fv = self.premises #np.array(self.premises).flatten()
    op  = input[self.end_x1:self.end_x2]
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    return error   

def goal_premises(input, self):
            
    fv = input[:self.end_x1].reshape(np.shape(self.premises))
    op  = self.op
    tsk = self.tsk
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    return error

def goal_operators(input, self):
            
    fv = self.premises #np.array(self.premises).flatten()
    op  = input[self.end_x1:self.end_x2]
    tsk = self.tsk
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    return error  
    
def goal_consequents(input, self):
            
    fv = self.premises
    op  = self.op
    tsk = input[self.end_x2:]
    new_labels = self.anfis_estimate_labels(fv, op, tsk)
    
    error = (np.abs(new_labels - self.expected_labels) ).sum()
    return error