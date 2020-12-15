"""
ID : 31431127

This script completed task 1 and 4. Also it contains necessary function defined by myself.
"""

import numpy as np
import scipy.optimize

# Task 1
def MyRK3_step(f, t, qn, dt, options=None):
    """
    This function is to compute s single step by using explicit RK3 method. t and qn shows the current situation.

    Parameters
    ----------
    f : function
        The function that defines the ODE
    t : float
        The time tn
    qn : array of float
        The initial data
    dt : float
        The time step
    options : dict
        Default ot none. Additional parameters defining the right hand side

    Returns
    -------
    qn_new : array of float
        2 elements array. The updated qn after a single step.
    """
    assert (hasattr(f, '__call__')), 'f must be a callable function'
    assert (np.isfinite(t) and (not np.isnan(t)) and np.isreal(t) and (t >= 0.)), 't must be finite, real and positive'
    assert (not np.any(np.isnan(qn)) and np.all(np.isfinite(qn)) and \
            np.all(np.isreal(qn))), 'qn must be finite and real '
    assert (np.isfinite(dt) and (not np.isnan(dt)) and np.isreal(dt) and \
            (dt >= 0.)), 'dt must be finite, real and positive'
    assert ((type(options) is dict) or (options is None)), 'options must be a dictionary or None '
    # To guarantee that qn is in the shape we want
    qn = np.array(qn).reshape(-1, )
    # Compute relevant k
    k_1 = f(t, qn, options)
    k_2 = f(t + 1 / 2 * dt, qn + dt * 1 / 2 * k_1, options)
    k_3 = f(t + dt, qn + dt * (-1 * k_1 + 2 * k_2), options)
    # Update qn by using previous calculated k_1,k_2,k_3
    qn_new = qn + dt * 1 / 6 * (k_1 + 4 * k_2 + k_3)

    assert (len(qn_new) == len(qn)), 'New qn must have the same number of elements as qn'
    return qn_new


# Task 4
def MyGRRK3_step(f, t, qn, dt, options=None):
    """
    This function is to compute a single step of qn by using GRRK3 method

    Parameters
    ----------
    f : function
        The RHS function of the equation
    t : float
        The time
    qn : array of float
        Initial data qn
    dt : float
        Time step
    options : dict
        The dictionary that contains other relevant parameters

    Returns
    -------
    qn_new : array of float
        The updated qn given previous information of qn

    Notes : Commented lines suitable for format np.array([[],[]]) i.e column vector
    """
    assert (hasattr(f, '__call__')), 'f must be a callable function'
    assert (np.isfinite(t) and (not np.isnan(t)) and np.isreal(t) and (t >= 0.)), 't must be finite, real and positive'
    assert (not np.any(np.isnan(qn)) and np.all(np.isfinite(qn)) and \
            np.all(np.isreal(qn))), 'qn must be finite and real '
    assert (np.isfinite(dt) and (not np.isnan(dt)) and np.isreal(dt) and \
            (dt >= 0.)), 'dt must be finite, real and positive'
    assert ((type(options) is dict) or (options is None)), 'options must be a dictionary or None '

    # To guarantee that qn is in the shape we want
    qn = np.array(qn).reshape(-1, )
    # Initial guess
    k1 = f(t + dt / 3, qn, options)
    k2 = f(t + dt, qn, options)
    K_initial = np.vstack((k1, k2))

    def F(K):
        """
        This function is to compute the result of the RHS function defined in equation (7)

        Parameters
        K: array of float
            An array contains k1 and k2

        Returns
        -------
        root5 : array of float
            The result of the RHS of the function
        """
        # Reshape K for later processes as the special input array in scipy root function
        K = K.reshape(2, -1)  # len(qn)
        k_1, k_2 = K[0], K[1]
        # Construct the array
        right_top = f(t + 1 / 3 * dt, qn + dt / 12 * (5 * k_1 - k_2), options)
        right_bot = f(t + dt, qn + dt / 4 * (3 * k_1 + k_2), options)
        right = np.vstack((right_top, right_bot))
        # RHS
        root = K - right
        root = root.reshape(-1, )
        return root

    root = scipy.optimize.root(F, K_initial).x
    k1_new, k2_new = root.reshape(2, -1)  # To separate
    # Compute the new qn
    qn_new = qn + dt / 4 * (3 * k1_new + k2_new)
    assert (len(qn_new) == len(qn)), 'Updated qn must have the number of elements of qn '
    return qn_new


def Runge_Kutta3(step_method, f, t, qn, dt, t_end, options=None):
    """
    This function is to compute numerical solution at every step by iterating for N times by using
    step_func(RK3 or GRRK3). Given the end of point of t and dt, N can be computed.

    Parameters
    ----------
    step_method : function
        The function which computes the qn+1 based on the previous information qn
    f : function
        The right-hand-side of the ODE
    t : float
        Initial time
    qn : array of float
        numpy array. Initial data
    dt : float
        The step size
    t_end : int
        The end point of t
    options : dict
        Other relevant parameters of the ODE

    Returns
    -------
    qn_list : array of float
        Values of computed qn

    Notes : The initial value is also included in the qn_list
    """
    assert (hasattr(step_method, '__call__')), 'step_method must be a callable function'
    assert (hasattr(f, '__call__')), 'step_method must be a callable function'
    assert (np.isfinite(t) and (not np.isnan(t)) and np.isreal(t) and (t >= 0.)), 't must be finite, real and positive'
    assert (not np.any(np.isnan(qn)) and np.all(np.isfinite(qn)) and \
            np.all(np.isreal(qn))), 'qn must be finite and real '
    assert (np.isfinite(dt) and (not np.isnan(dt)) and np.isreal(dt) and \
            (dt >= 0.)), 'dt must be finite, real and positve'
    assert ((type(options) is dict) or (options is None)), 'options must be a dictionary or None '

    # To guarantee that qn is in the shape we want
    qn = np.array(qn).reshape(-1, )
    # Generate integer N （loop times)
    N = int((t_end - t) / dt)
    assert type(N) is int, 'Should be integer'

    # Initialize the list of qn
    qn_list = np.zeros((N + 1, len(qn)))
    qn_list[0] = qn
    # Loop N times
    for n in range(N):
        qn = step_method(f, t, qn, dt, options)
        qn_list[n + 1] = qn
        t += dt  # Update t for next iteration
    return qn_list


# The RHS of the equation (2) defined in the cwk1
def System_function(t, q, options):
    """
    This the function which defined in coursework 1 equation (2)

    Parameters
    ----------
    t : float
        The time
    q : array of float
        qn data
    options : dict
        The dictionary containing other necessary parameters

    """
    assert (np.isfinite(t) and (not np.isnan(t)) and np.isreal(t) and (t >= 0.)), 't must be finite, real and positive'
    assert (not np.any(np.isnan(q)) and np.all(np.isfinite(q)) and \
            np.all(np.isreal(q))), 'qn must be finite and real '
    assert ((type(options) is dict) or (options is None)), 'options must be a dictionary or None '
    # Fetch parameters from options argument
    gamma = options.get('gamma')
    w = options.get('omega')
    epsilon = options.get('epsilon')
    # Construct parameter matrix
    parameter_matrix = np.array([[gamma, epsilon], [epsilon, -1]])

    # Construct the left matrix in the function
    x, y = q[0], q[1]
    middle_matrix = np.array([[(-1 + x ** 2 - np.cos(t)) / (2 * x)],
                              [(-2 + y ** 2 - np.cos(w * t)) / (2 * y)]])
    right_matrix = np.array([[np.sin(t) / (2 * x)],
                             [w * np.sin(w * t) / (2 * y)]])
    # Compute result
    result_sys = np.dot(parameter_matrix, middle_matrix) - right_matrix
    result_sys = result_sys.reshape(-1, )  # Guarantee the shape of the returned vector
    return result_sys
