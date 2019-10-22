import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
import time

def adam_optimize(objective, params, jac, step_size=1e-2, Nsteps=100, bounds=None, 
                    options={'direction': 'min'}):
    """Performs Nsteps steps of ADAM minimization of function `objective` with gradient `jac`.
    The `bounds` are set abruptly by rejecting an update step out of bounds."""
    of_list = []

    opt_keys = options.keys()
    np.set_printoptions(formatter={'float': '{: 1.4f}'.format})

    if 'beta1' in opt_keys:
        beta1 = options['beta1']
    else:
        beta1 = 0.9

    if 'beta2' in opt_keys:
        beta2 = options['beta2']
    else:
        beta2 = 0.999

    for iteration in range(Nsteps):

        t_start = time.time()
        if jac==True:
            of, grad = objective(params)
        else:
            of = objective(params)
            grad = jac(params)
        t_elapsed = time.time() - t_start

        of_list.append(of._value if type(of) is ArrayBox else of) 

        disp_str = "Epoch: %3d/%3d | Duration: %.2f secs" % (iteration+1, Nsteps, t_elapsed)
        if 'disp' in opt_keys:
            if 'of' in options['disp']:
                disp_str += " | Value: %5e" % (of_list[-1])
            if 'params' in options['disp']:
                disp_str += " | Parameters: %s" % params
        print(disp_str)

        if iteration == 0:
            mopt = np.zeros(grad.shape)
            vopt = np.zeros(grad.shape)

        (grad_adam, mopt, vopt) = step_adam(grad, mopt, vopt, iteration, beta1, beta2)

        if options['direction'] == 'min':
            params = params - step_size*grad_adam
        elif options['direction'] == 'max':
            params = params + step_size*grad_adam
        else:
            print("options[direction] should be either 'min' or 'max'")

        if bounds:
            params[params < bounds[0]] = bounds[0]
            params[params > bounds[1]] = bounds[1]

    return (params, of_list)


def step_adam(gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
    """ Performs one step of adam optimization"""

    mopt = beta1 * mopt_old + (1 - beta1) * gradient
    mopt_t = mopt / (1 - beta1**(iteration + 1))
    vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
    vopt_t = vopt / (1 - beta2**(iteration + 1))
    grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

    return (grad_adam, mopt, vopt)
