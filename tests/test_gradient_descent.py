import sys, os
import math
import numpy as np

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from algorithms import gradient_descent

EPSILON = 1e-3

def test_gradient_with_x2_and_zero():
    assert abs(gradient_descent.gradient(lambda x: x**2, 0.0)) <= EPSILON

def test_gradient_with_x2_and_2():
    assert abs(gradient_descent.gradient(lambda x: x**2, 2.0) - 4.0) <= EPSILON

def test_gradient_with_3_variables():
    x0 = [2.0, 3.0, math.pi/3]
    grad_calc = gradient_descent.gradient(lambda x,y,z: 2*x + 3*(y**2) - math.sin(z), x0)
    grad_true = [2.0, 6*x0[1], - math.cos(x0[2])]
    assert max([abs(g_c - g_t) for g_c, g_t in zip(grad_calc, grad_true)]) <= EPSILON

def test_gradient_descent_for_function_with_one_variable():
    f = lambda x: x**4 - 3*(x**3)+2
    x_true = 9.0 / 4.0
    min_true = f(x_true)
    gd_calc = gradient_descent.gradient_descent_for_function(f, x0=6.0, step_size=0.01)
    assert max(abs(gd_calc["f"] - min_true), abs(gd_calc["x"] - x_true)) <= EPSILON