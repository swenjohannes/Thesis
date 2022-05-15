# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:00:19 2022

@author: Swen
"""
from time import time

def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f' \t Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
