# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:17:43 2022

@author: arka
"""

import numpy as np
import matplotlib.pyplot as plt

def func_1(x):
    return x+3

def func_1_dd(x):
    return np.zeros(x.shape)

def func_2(x):
    return np.sin(2 * np.pi * x)

def func_2_dd(x):
    return -4 * np.pi * np.pi * func_2(x)

def func_3(x):
    return 5*(x**4) + 3*(x**2) + 6*x + 2

def func_3_dd(x):
    return 60*(x**2) + 6


def compute(f, f_dd, scheme, lower, upper, error_scheme, points_count=10):
    unit = (upper - lower) / points_count
    points = np.arange(lower, upper + unit, unit)
    f_vals = f(points)
    
    if scheme == '2-CDS':
        res = f_dd(points[1:-1])
        kernel = np.array([1, -2, 1])
        dividend = unit ** 2
    elif scheme == '4-CDS':
        res = f_dd(points[2:-2])
        kernel = np.array([-1, 16, -30, 16, -1])
        dividend = 12 * (unit ** 2)
    elif scheme == '3-FDS':
        res = f_dd(points[:-4])
        kernel = np.array([35, -104, 114, -56, 11])
        dividend = 12 * (unit ** 2)

        
    discrete_res = np.convolve(f_vals, kernel, 'valid') / dividend
    internal_points = len(discrete_res)

    
    if error_scheme == 'L-inf':
        error = max(np.abs(res - discrete_res))
    elif error_scheme == 'rms':
        error = np.linalg.norm(res-discrete_res) / np.sqrt(internal_points)
        
    return error
    


upper = 1
lower = 0
scales = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001][::-1]
schemes = ['2-CDS', '3-FDS', '4-CDS']
error_schemes = ['L-inf', 'rms']
functions = [
        (1, func_1, func_1_dd),
        (2, func_2, func_2_dd),
        (3, func_3, func_3_dd)
    ]
function_names = {
    1 : 'x + 3',
    2 : 'sin(2πx)',
    3 : '5x⁴ + 3x² + 6x + 2'    
}

res = {}

for index, f, f_dd in functions:
    error_scheme_data = {}
    for error_scheme in error_schemes:
        scheme_data = {}
        for scheme in schemes:
            errors = []
            for scale in scales:
                err = compute(f, f_dd, scheme, lower, upper, error_scheme, (upper - lower) / scale)
                errors.append(err)
            scheme_data[scheme] = np.log10(errors)
        error_scheme_data[error_scheme] = scheme_data
    res[index] = error_scheme_data
            
for index, data in res.items():
    errors = {}
    
    for error_scheme in error_schemes:
        for scheme in schemes:
            errors[error_scheme + " " + scheme] = data[error_scheme][scheme]
            
    x = -np.log10(scales)
    plt.figure()
    plt.title(function_names[index])
    for plot, y in errors.items():
        plt.plot(x, y, label=plot, marker='o')
    plt.ylabel("log₁₀(error)")
    plt.xlabel("-log₁₀(Δx)")
    plt.legend()
    plt.show()
            
                