import numpy as np
from scipy.optimize import minimize
from sympy import *


def profit(x, price, func, w_k_func, w_l_func):
    k_val, l_val = x
    return price * func.subs({k: k_val, l: l_val}) - (w_k_func * k_val + w_l_func * l_val)


def max_for_minimize(x, price, func, w_k_func, w_l_func):
    return -profit(x, price, func, w_k_func, w_l_func)


def p_monopoly(q):
    return 800000 - 4*q


def w_k_monopsony(k_val):
    return 20000 + 2 * k_val


def w_l_monopsony(l_val):
    return 17000 + l_val


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
    print()

    # Optimal factors under conditions of perfect competition
    p = 7
    w_k = 3
    w_l = 5

    bounds = [(0, None), (0, None)]
    result = minimize(max_for_minimize, x0=np.array([30000, 30000]), args=(p, F_func, w_k, w_l), bounds=bounds)
    K_opt_long, L_opt_long = result.x
    print("Long-term period:")
    print("K*, L*: ", K_opt_long.round(5), L_opt_long.round(5))
    print("Profit: ", profit([K_opt_long, L_opt_long], p, F_func, w_k, w_l))
    print()

    constraints = ({'type': 'ineq',
                    'fun': lambda x: 30000 - x[0] - x[1]})
    result = minimize(max_for_minimize, x0=np.array([30000, 30000]), args=(p, F_func, w_k, w_l), constraints=constraints)
    K_opt_short, L_opt_short = result.x
    print("Short-term period")
    print("K*, L*: ", K_opt_short.round(5), L_opt_short.round(5))
    print("Profit: ", profit([K_opt_short, L_opt_short], p, F_func, w_k, w_l))
    print()

    #optimal factors under monopoly-monopsony conditions
    result = minimize(lambda x: max_for_minimize(x, p_monopoly(F_func.subs({k: x[0], l: x[1]})), F_func,
                                                 w_k_monopsony(x[0]), w_l_monopsony(x[1])), x0=np.array([20000, 20000]),
                      bounds=bounds)
    K_opt_mono, L_opt_mono = result.x
    print("term period")
    print("K*, L*: ", K_opt_mono.round(5), L_opt_mono.round(5))
    w_l_monopsony = w_l_monopsony(L_opt_mono)
    w_k_monopsony = w_k_monopsony(K_opt_mono)
    q = F_func.subs({k: K_opt_mono, l: L_opt_mono})
    price = p_monopoly(q)
    print("Price K: ", w_k_monopsony.round(5))
    print("Price L: ", w_l_monopsony.round(5))
    print("Price: ", price.round(5))
    print("Quantity: ", q)
    print("Profit: ", profit([K_opt_mono, L_opt_mono], price, F_func,
                                                 w_k_monopsony, w_l_monopsony))
