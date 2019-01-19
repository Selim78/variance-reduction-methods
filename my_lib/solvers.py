import numpy as np

def GD(w0, grad, prox, max_iter):
    n, L      = grad.n, grad.L
    stepsize  = 1/L
    w         = w0
    w_tab     = np.copy(w)

    for k in range(max_iter):
        w     = prox(w - stepsize*grad(w), stepsize)
        w_tab = np.vstack((w_tab, w))

    return w, w_tab

def SGD(w0, grad, prox, max_iter):
    n, L  = grad.n, grad.L
    w     = w0
    w_tab = np.copy(w)

    for k in range(max_iter):
        stepsize  = 1.0/ (L*(1+k/10.0)**0.6)
        i         = np.random.choice(n)
        w         = prox(w - stepsize*grad(w, i), stepsize)
        if k%n == 0: # each completed epoch
            w_tab = np.vstack((w_tab, w))

    return w, w_tab

def SAG(w0, grad, prox, max_iter):
    n, L, mu      = grad.n, grad.L, grad.mu
    stepsize      = 1.0/(16.0*L)
    w             = w0
    w_tab         = np.copy(w)
    gradients     = [grad(w, i) for i in range(n)]

    for k in range(max_iter):
        full_gradient = sum(gradients)/n
        i             = np.random.choice(n)
        gradient_i_k  = grad(w, i)
        w             = prox(w - stepsize*((gradient_i_k - gradients[i])/n + full_gradient), stepsize)
        gradients[i]  = gradient_i_k
        if k%grad.n == 0: # each completed epoch
            w_tab = np.vstack((w_tab, w))

    return w, w_tab

def SAGA(w0, grad, prox, max_iter):
    n, L, mu      = grad.n, grad.L, grad.mu
    stepsize      = 1.0/(2.0*(L+mu*n))
    w             = w0
    w_tab         = np.copy(w)
    gradients     = [grad(w, i) for i in range(n)]

    for k in range(max_iter):
        full_gradient = sum(gradients)/n
        i             = np.random.choice(n)
        gradient_i_k  = grad(w, i)
        w             = prox(w - stepsize*(gradient_i_k - gradients[i] + full_gradient), stepsize)
        gradients[i]  = gradient_i_k
        if k%grad.n == 0: # each completed epoch
            w_tab = np.vstack((w_tab, w))

    return w, w_tab


def SVRG(w0, grad, prox, max_iter):
    n, L, mu = grad.n, grad.L, grad.mu
    stepsize = 1.0/L
    w        = w0
    w_tab    = np.copy(w)
    M        = 10 # inner loop

    for k in range(max_iter):
        gradients     = [grad(w, i) for i in range(n)]
        full_gradient = sum(gradients)/n
        w_k           = w

        for j in range(M):
            i   = np.random.choice(n)
            w_k = prox(w_k - stepsize*(grad(w_k, i) + full_gradient - gradients[i]), stepsize)

        w = w_k

        w_tab = np.vstack((w_tab, w))

    return w, w_tab

# _______________________ parallel ________________________

from os import cpu_count
from joblib import Parallel, delayed

def hogwildSGD(w0, grad, prox, max_iter):
    n, L  = grad.n, grad.L

    # Shared accross the parallel jobs
    shared = {'k':0, 'w':w0, 'w_tab':np.copy(w0)}

    def step():
        stepsize    = 1.0/ (L*(1+shared['k']/10.0)**0.6)
        i           = np.random.choice(n)
        shared['w'] = prox(shared['w'] - stepsize*grad(shared['w'], i), stepsize)
        if shared['k']%n == 0: # each completed epoch
            shared['w_tab'] = np.vstack((shared['w_tab'], shared['w']))
        shared['k'] += 1

    # Number of jobs chosen as the number of CPU detected
    n_jobs = cpu_count()
    print("Detected CPU: ", n_jobs)
    Parallel(n_jobs=20, require='sharedmem', prefer="threads")(delayed(step)() for _ in range(max_iter))

    return shared['w'], shared['w_tab']
