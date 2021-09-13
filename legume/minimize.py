import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd import value_and_grad
from scipy.optimize import minimize
import time


class Minimize(object):
    """Wrapping up custom and SciPy optimizers in a common class
    """
    def __init__(self, objective):

        self.objective = objective

        # Some internal variables
        self.iteration = 0
        self.of_list = []
        self.p_opt = []
        self.t_store = time.time()

    @staticmethod
    def _get_value(x):
        """This is used when gradients are computed with autograd and the 
        objective function is an ArrayBox. Same function as in legume.utils, 
        but re-defined here so that this class could also be used independently 
        """
        if str(type(x)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
            return x._value
        else:
            return x

    def _parse_bounds(self, bounds):
        """Parse the input bounds, which can be 'None', a list with two
        elements, or a list of tuples with 2 elements each
        """
        try:
            if bounds == None:
                return None
            elif not isinstance(bounds[0], tuple):
                if len(bounds) == 2:
                    return [tuple(bounds) for i in range(self.params.size)]
                else:
                    raise ValueError
            elif len(bounds) == self.params.size:
                if all([len(b) == 2 for b in bounds]):
                    return bounds
                else:
                    raise ValueError
            else:
                raise ValueError
        except:
            raise ValueError(
                "'bounds' should be a list of two elements "
                "[lb, ub], or a list of the same length as the number of "
                "parameters where each element is a tuple (lb, ub)")

    def _disp(self, t_elapsed):
        """Display information at every iteration
        """
        disp_str = "Epoch: %4d/%4d | Duration: %6.2f secs" % \
                        (self.iteration, self.Nepochs, t_elapsed)
        disp_str += " | Objective: %4e" % self.of_list[-1]
        if self.disp_p:
            disp_str += " | Parameters: %s" % self.params
        print(disp_str)

    def adam(self,
             pstart,
             Nepochs=50,
             bounds=None,
             disp_p=False,
             step_size=1e-2,
             beta1=0.9,
             beta2=0.999,
             args=(),
             pass_self=False,
             callback=None):
        """Performs 'Nepoch' steps of ADAM minimization with parameters 
        'step_size', 'beta1', 'beta2'

        Additional arguments:
        bounds          -- can be 'None', a list of two elements, or a 
            scipy.minimize-like list of tuples each containing two elements
            The 'bounds' are set abruptly after the update step by snapping the 
            parameters that lie outside to the bounds value
        disp_p          -- if True, the current parameters are displayed at 
            every iteration
        args            -- extra arguments passed to the objective function
        pass_self       -- if True, then the objective function should take
            of(params, args, opt), where opt is an instance of the Minimize 
            class defined here. Useful for scheduling
        Callback        -- function to call at every epoch; the argument that's
                            passed in is the current minimizer state
        """
        self.params = pstart
        self.bounds = self._parse_bounds(bounds)
        self.Nepochs = Nepochs
        self.disp_p = disp_p

        # Restart the counters
        self.iteration = 0
        self.t_store = time.time()
        self.of_list = []

        if pass_self == True:
            arglist = list(args)
            arglist.append(self)
            args = tuple(arglist)

        for iteration in range(Nepochs):
            self.iteration += 1

            self.t_store = time.time()
            of, grad = value_and_grad(self.objective)(self.params, *args)
            t_elapsed = time.time() - self.t_store

            self.of_list.append(self._get_value(of))
            self._disp(t_elapsed)

            if iteration == 0:
                mopt = np.zeros(grad.shape)
                vopt = np.zeros(grad.shape)

            (grad_adam, mopt, vopt) = self._step_adam(grad, mopt, vopt,
                                                      iteration, beta1, beta2)
            # Change parameters towards minimizing the objective
            if iteration < Nepochs - 1:
                self.params = self.params - step_size * grad_adam

            if bounds:
                lbs = np.array([b[0] for b in self.bounds])
                ubs = np.array([b[1] for b in self.bounds])
                self.params[self.params < lbs] = lbs[self.params < lbs]
                self.params[self.params > ubs] = ubs[self.params > ubs]

            if callback is not None:
                callback(self)

        return (self.params, self.of_list)

    @staticmethod
    def _step_adam(gradient,
                   mopt_old,
                   vopt_old,
                   iteration,
                   beta1,
                   beta2,
                   epsilon=1e-8):
        """Performs one step of Adam optimization
        """

        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1**(iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
        vopt_t = vopt / (1 - beta2**(iteration + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)

    def lbfgs(self,
              pstart,
              Nepochs=50,
              bounds=None,
              disp_p=False,
              maxfun=15000,
              args=(),
              pass_self=False,
              res=False,
              callback=None):
        """Wraps the SciPy LBFGS minimizer in a way that displays intermediate
        information and stores intermediate values of the parameters and the
        objective function.

        Nepochs         -- Maximum number of iterations
        bounds          -- can be 'None', a list of two elements, or a 
            scipy.minimize-like list of tuples each containing two elements
            The 'bounds' are set abruptly after the update step by snapping the 
            parameters that lie outside to the bounds value
        disp_p          -- if True, the current parameters are displayed at 
            every iteration
        maxfun          -- Maximum number of function evaluations
        args            -- extra arguments passed to the objective function
        pass_self       -- if True, then the objective function should take
            of(params, args, opt), where opt is an instance of the Minimize 
            class defined here. Useful for scheduling
        res             -- if True, will also return the SciPy OptimizeResult
        callback        -- function to call at every epoch; the argument that's
                            passed in is the current minimizer state
        """

        self.params = pstart
        self.bounds = self._parse_bounds(bounds)
        self.Nepochs = Nepochs
        self.disp_p = disp_p

        # Restart the counters
        self.iteration = 0
        self.t_store = time.time()
        self.of_list = []

        # Get initial of value
        of = self.objective(self.params, *args)
        self.of_list.append(self._get_value(of))

        def of(params, *args, **kwargs):
            """Modify the objective function slightly to allow storing
            intermediate objective values without re-evaluating the function
            """
            if pass_self == True:
                arglist = list(args)
                arglist.append(self)
                args = tuple(arglist)

            out = value_and_grad(self.objective)(params, *args, **kwargs)
            self.of_last = self._get_value(out[0])
            return out

        def cb(xk):
            """Callback function for the SciPy minimizer
            """
            self.iteration += 1
            t_current = time.time()
            t_elapsed = t_current - self.t_store
            self.t_store = t_current

            self.of_list.append(self.of_last)
            self.params = xk
            self._disp(t_elapsed)

            # Call the custom callback function if any
            if callback is not None:
                callback(self)

        res_opt = minimize(of,
                           self.params,
                           args=args,
                           method='L-BFGS-B',
                           jac=True,
                           bounds=self.bounds,
                           tol=None,
                           callback=cb,
                           options={
                               'disp': False,
                               'maxcor': 10,
                               'ftol': 1e-8,
                               'gtol': 1e-5,
                               'eps': 1e-08,
                               'maxfun': maxfun,
                               'maxiter': Nepochs,
                               'iprint': -1,
                               'maxls': 20
                           })

        if res == False:
            return (res_opt.x, self.of_list)
        else:
            return (res_opt.x, self.of_list, res_opt)
