# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:53:48 2022

@author: Swen
"""

import functools

def repeat(nbrTimes=2):
    '''
    Define parametrized decorator with arguments
    Default nbr of repeats is 2
    '''
    def real_repeat(func):
        """
        Repeats execution 'nbrTimes' times
        """
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(nbrTimes):
                func(*args, **kwargs)
        return wrapper_repeat
    return real_repeat
