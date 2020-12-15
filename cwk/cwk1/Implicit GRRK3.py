import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

"""
Question 4
"""


def MyGRRK3_step(f, t, qn, dt, options=None):
    """

    Comment lines suitable for np.array([[0],[0]])
    """
    k1 = f(t + dt / 3, qn, options=None)  # .T
    k2 = f(t + dt, qn, options=None)  # .T
    K_initial = np.vstack((k1, k2))

    def F(K):
        K = K.reshape(len(qn), -1)
        k_1, k_2 = K[0], K[1]
        # k_1, k_2 = K[0].reshape(-1, 1), K[1].reshape(-1, 1)
        # Construct the array
        right_top = f(t + 1 / 3 * dt, qn + dt / 12 * (5 * k_1 - k_2), options=None)  # .T
        right_bot = f(t + dt, qn + dt / 4 * (3 * k_1 + k_2), options=None)  # .T
        right = np.vstack((right_top, right_bot))
        # RHS
        root5 = K - right
        root5 = root5.reshape(-1, )
        return root5

    root5 = scipy.optimize.root(F, K_initial).x
    k1_new, k2_new = root5.reshape(len(qn), -1)
    # k1_new = k1_new.reshape(-1, 1)
    # k2_new = k2_new.reshape(-1, 1)
    # Compute the new qn
    qn_new = qn + dt / 4 * (3 * k1_new + k2_new)
    return qn_new


def f_5(t, q, options=None):
    """
    The RHS of the function defined in question 5

    Parameters
    ----------
    t : float
        Time
    q : array of float
        qn
    options : dict
        Other parameters,default to none

    Returns
    -------
    result_5 : array of float
        2D array. The function value

    """
    result_5 = np.zeros_like(q)
    result_5[0] = q[1] + 1
    result_5[1] = 6 * t
    return result_5


def q5_error_vanish(t_end=0.1):
    qn_5 = np.array([0, 0])
    truncated = MyGRRK3_step(f_5, 0, qn_5, t_end)
    real = np.array([t_end ** 3 + t_end, 3 * t_end ** 2])
    error = np.abs(truncated-real)
    if np.allclose(truncated, real) is True:
        return 'Error vanishes. Proof completed. Error is {}'.format(error)
    else:
        return 'Error still exists'
