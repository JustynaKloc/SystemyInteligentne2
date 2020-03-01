# -*- coding: utf-8 -*-
import time
import typing

# Matlab - tic oraz toc

def tic(name: str = None):
    global __tic_value, __tic_name
    __tic_value = time.time()
    __tic_name = name

def toc():
    if '__tic_value' not in globals():
        print("Brak tic???")
        return
    
    print('\x1b[35m\x1b[1m')
    print(f"Czas operacji [{__tic_name}]: {time.time() - __tic_value}sek")
    print('\x1b[0m')
    

        

