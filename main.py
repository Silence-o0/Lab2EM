import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve, minimize
from sympy import *


def profit(x):
    k_val, l_val = x
    return (p*(b0*k_val**b1*l_val**b2)-(w_k*k_val+w_l*l_val))


def max_for_minimize(x):
    return -profit(x)


if __name__ == '__main__':
    F = [140330, 120355, 125000, 137330, 121570, 113100, 133000, 126165, 149000, 120950]
    K = [5700, 4740, 4390, 5330, 5200, 4160, 5200, 4690, 5890, 4930]
    L = [12245, 13340, 13860, 14400, 14000, 11000, 14145, 13900, 15050, 13060]
    observations = len(F)

    init_printing(use_unicode=False, wrap_line=False)
    k = Symbol('k')
    l = Symbol('l')

    # The least-squares solution
    # F = b0 * K^b1 * L^b2
    # ln(F) = ln(b0) + b1*ln(K) + b2*ln(L)
    b0, b1, b2 = np.linalg.lstsq(np.vstack([np.ones(len(K)), np.log(K), np.log(L)]).T, np.log(F), rcond=None)[0]
    b0 = np.exp(b0)
    F_func = b0 * k**b1 * l**b2
    print(F_func)

    # Scale effect
    scale = 3
    print("A * F(K,L);   F(A*K, A*L)")
    for i in range(len(F)):
        f_scale = F_func.subs({k: scale*K[i], l: scale*L[i]})
        mult_f_scale = scale * F_func.subs({k: K[i], l: L[i]})
        print(f"{i+1}: {mult_f_scale}  {f_scale}")
        # Increasing returns to scale

    # Elasticity of substitution
    print("Elasticity of substitution:")
    t1 = Symbol('t1')
    t2 = Symbol('t2')

    a = ln(k/l)
    MP_k = diff(F_func, k)
    MP_l = diff(F_func, l)
    b = ln(MP_k/MP_l)

    a_sub = a/a*t1
    b_coef = exp(b)/(1/exp(a))
    print(f"-(d({a})/d(log(MP_k/MP_l)))  = -(d({a})/d({b}))  =  -(d({a_sub})/d({ln(b_coef)-t1}))  =  "
          f"\n-(d({a_sub.subs(t1, (ln(b_coef)-t2))})/d({t2}))  =  {-diff(a_sub.subs(t1, (ln(b_coef)-t2)), t2)}")

    # Optimal factor under conditions of perfect competition
    p = 9
    w_k = 2
    w_l = 3

    bounds = [(0, None), (0, None)]
    result = minimize(max_for_minimize, [5000, 13000], bounds=bounds)
    K_opt, L_opt = result.x
    print(K_opt, L_opt)
    print(profit([K_opt, L_opt]))


    constraints = ({'type': 'ineq', 'fun': lambda x: 30000 - x[0] - x[1]})
    result = minimize(max_for_minimize, [5000, 13000], constraints=constraints)
    K_opt, L_opt = result.x
    print(K_opt, L_opt)
    print(profit([K_opt, L_opt]))




