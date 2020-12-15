"""
ID:31431127

This script is the master file which produces all plots required by the coursework.
The other script contains task 1,4 and necessary self-defined functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from CWK_func_31431127 import MyRK3_step
from CWK_func_31431127 import MyGRRK3_step
from CWK_func_31431127 import Runge_Kutta3
from CWK_func_31431127 import System_function
import time


# Task 2
def error_vanish(Step_func, dt=0.1):
    """
    This function is to prove that error vanishes at a specific point after a simple step

    Parameters
    ----------
    Step_func ：function
        The step function that used in the unit test

    dt : float
        The step size of t. In this case, dt is also the end of t

    Returns
    -------

    The result of whether the error vanishes after one step

    """
    assert (np.isfinite(dt) and (not np.isnan(dt)) and np.isreal(dt) and \
            (dt >= 0.)), 'dt must be finite, real and positive'

    def f_2(t, q, options=None):
        """
        The RHS of the function defined in question (2)

        Parameters
        ----------

        t : float
            Time point
        q : array of float
            qn
        options : dict
            Other parameters, default set to none

        Returns
        -------

        result_2 : array of float
            2D array. The function value.
        """
        result_2 = np.zeros_like(q)
        result_2[0] = q[1] + 1
        result_2[1] = 6 * t
        return result_2

    q0 = np.array([0, 0])
    # Numerical solution after one step
    truncation = Step_func(f_2, 0, q0, dt)
    # Exact solution
    real = np.array([dt ** 3 + dt, 3 * dt ** 2])
    # print(real-truncation)
    assert np.allclose(truncation, real), 'Error still exists'


# Task 3/5
def Plot_convergence_rate(step_func, method_name, t_end=0.5):  # single step
    """
    This function is to plot the convergence rate of explicit RK3 method

    Parameters
    ----------
    step_func : function
        The step method used to check the convergence rate
    method_name : str
        The name of the algorithm. Used in title of the plot

    t_end: float
        The point that we use to control the step size. Default to 0.1


    Returns
    -------
    A plot shows the convergence rate

    """

    # Define the RHS of the function
    def f3(t, q, options=None):
        """
        This function is RHS of the ODE defined in question 3

        Parameters
        ---------
        t : float
            Time
        q : array of float
            Point qn
        options : dict
            Other parameters

        Returns
        -------
        result_3 : float
            Result of the defined function

        """
        result_3 = t - q
        return result_3

    assert (hasattr(step_func, '__call__')), 'step_func must be a callable function'
    assert (isinstance(method_name, str)), 'method_name must be a string '
    assert (np.isfinite(t_end) and (not np.isnan(t_end)) and np.isreal(t_end) and \
            (t_end >= 0.)), 't_end must be finite, real and positive'

    # Generate a list of N
    Npoints = np.array([2 ** i for i in range(10)])
    # Determine the step size
    dt_3 = t_end / Npoints
    # Initialize the list of error
    error_3 = np.zeros_like(dt_3)
    # Initialize the point qn
    qn_3 = np.array([0], dtype=float)
    # The ture result of qn at point x_end
    Exact_result = lambda t_end: np.exp(-1 * t_end) + t_end - 1

    # Compute the error
    for i, dt in enumerate(dt_3):  # len（dt_3)-1
        numerical_result = step_func(f3, 0., qn_3, dt)
        error_3[i] = np.abs(numerical_result - Exact_result(dt))

    # Fit the points with a straight line
    p_3 = np.polyfit(np.log(dt_3), np.log(error_3), 1)

    # Plot the required convergence rate
    plt.loglog(dt_3, error_3, 'kx', label='Numerical Data')
    plt.loglog(dt_3, np.exp(p_3[1]) * dt_3 ** p_3[0], "b-", label='Best Fit Line slope {:.2f}'.format(p_3[0]))
    plt.xlabel(r'$dt$')
    plt.ylabel(r'$|Error|$')
    plt.title(
        'Task:{} Convergence Rate for {} single step method'.format(3 if method_name == 'RK3' else 5, method_name))
    plt.legend()
    plt.tight_layout()
    plt.show()


# Task 6
def Plot_nonstiff_rk3_grrk3():
    """
    This function is the required plot of question 6

    Returns
    -------
    A figure which shows the numerical results of two algorithms and the exact result
    """
    # The parameters in equation 2
    q6_parameters = {'gamma': -2, 'omega': 5, 'epsilon': 0.05}
    # dt
    dt_6 = 0.05
    # t_end
    t_end = 1.
    # Compute evenly spaced points
    t_list = np.linspace(0., t_end, int(t_end / dt_6) + 1)  # t_list = np.arange(0.0, 1.0 + dt_6, dt_6)
    # Initial qn data
    qn = np.array([2 ** 0.5, 3 ** 0.5])

    # Compute numerical results of both algorithms
    x_rk3_list = Runge_Kutta3(MyRK3_step, System_function, 0, qn, dt_6, t_end, options=q6_parameters)[:, 0]
    y_rk3_list = Runge_Kutta3(MyRK3_step, System_function, 0, qn, dt_6, t_end, options=q6_parameters)[:, 1]
    x_grrk3_list = Runge_Kutta3(MyGRRK3_step, System_function, 0, qn, dt_6, t_end, options=q6_parameters)[:, 0]
    y_grrk3_list = Runge_Kutta3(MyGRRK3_step, System_function, 0, qn, dt_6, t_end, options=q6_parameters)[:, 1]

    # The exact solution of the function
    exact_x = lambda t: np.sqrt(1 + np.cos(t))
    exact_y = lambda t: np.sqrt(2 + np.cos(q6_parameters['omega'] * t))

    # Initialize two subplots
    fig, axes = plt.subplots(1, 2)
    # LHS plot
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1.24, 1.42)
    axes[0].plot(t_list, x_rk3_list, 'kx', label='RK3')
    axes[0].plot(t_list, x_grrk3_list, 'r+', label='GRRK3')
    axes[0].plot(t_list, exact_x(t_list), label='Exact Solution', linewidth=1)
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$x$')
    # RHS plot
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 1.8)
    axes[1].plot(t_list, y_rk3_list, 'kx')
    axes[1].plot(t_list, y_grrk3_list, 'r+')
    axes[1].plot(t_list, exact_y(t_list), linewidth=1)
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$y$')
    fig.legend()
    fig.suptitle('Task 6: Numerical results of Non-stiff Problems')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Task 7
def Plot_nonstiff_convergence(t_end=0.1):
    """
    This function is to plot the convergence rate of both algorithms applied to non-stiff problem

    Parameters
    ----------
    t_end : float
        The end of t

    Returns
    -------
    Return a plot which has two subplots and the convergence rate of both algorithms are shown in the plot

    Notes : t_end can be changed. However, the convergence rate of both algorithms is approximately 3
    """
    assert (np.isfinite(t_end) and (not np.isnan(t_end)) and np.isreal(t_end) and \
            (t_end >= 0.)), 't_end must be finite, real and positive'
    # Initialize parameters
    q7_parameters = {'gamma': -2, 'omega': 5, 'epsilon': 0.05}
    # Construct the list of dt
    dt_ls = np.array([0.1 / 2 ** j for j in range(8)])
    # Initialize the list of 1-norm errors of both algorithms
    Error_rk3_ls = np.zeros_like(dt_ls)
    Error_grrk3_ls = np.zeros_like(dt_ls)
    # Exact y solution to the equation
    Exact_y = lambda t: np.sqrt(2 + np.cos(q7_parameters['omega'] * t))
    # Initial qn data
    qn = np.array([2 ** 0.5, 3 ** 0.5])

    # Loop in list of dt to compute every corresponding 1-norm error
    for i, dt in enumerate(dt_ls):
        # Generate the point of t_ls
        t_ls = np.linspace(0., t_end, int(t_end / dt) + 1)
        # Compute numerical solutions
        y_rk3_ls = Runge_Kutta3(MyRK3_step, System_function, 0, qn, dt, t_end, options=q7_parameters)[:, 1]  # List
        y_grrk3_ls = Runge_Kutta3(MyGRRK3_step, System_function, 0, qn, dt, t_end, options=q7_parameters)[:, 1]
        # Compute the 1-norm error
        Error_rk3 = dt * np.sum(np.abs(y_rk3_ls[1:] - Exact_y(t_ls[1:])))
        Error_rk3_ls[i] = Error_rk3
        Error_rk3 = dt * np.sum(np.abs(y_grrk3_ls[1:] - Exact_y(t_ls[1:])))
        Error_grrk3_ls[i] = Error_rk3

    # Use np.polyfit to fit the data
    p_rk3 = np.polyfit(np.log(dt_ls), np.log(Error_rk3_ls), 1)
    p_grrk3 = np.polyfit(np.log(dt_ls), np.log(Error_grrk3_ls), 1)

    # Plot required plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    # First subplot
    axes[0].loglog(dt_ls, Error_rk3_ls, 'kx', label='Numerical data')
    axes[0].plot(dt_ls, np.exp(p_rk3[1]) * dt_ls ** p_rk3[0], label='Best Fit Line slope{:.3f}'.format(p_rk3[0]))
    axes[0].legend()
    axes[0].set_xlabel(r'$dt$')
    axes[0].set_ylabel(r'$Error$')
    axes[0].set_title('RK3')
    # Second subplot
    axes[1].loglog(dt_ls, Error_grrk3_ls, 'kx', label='Numerical data')
    axes[1].plot(dt_ls, np.exp(p_grrk3[1]) * dt_ls ** p_grrk3[0], label='Best Fit Line slope{:.3f}'.format(p_grrk3[0]))
    axes[1].legend()
    axes[1].set_xlabel(r'$dt$')
    axes[1].set_ylabel(r'$Error$')
    axes[1].set_title('GRRK3')
    fig.suptitle('Task 7: Non-Stiff Convergence Rate')
    fig.tight_layout()
    plt.show()


# Task 8/9
def Plot_stiff_solution(Step_method, Step_name, dt, t_end, tlim, xlim, ylim):
    """
    This function is to plot the numerical and exact solutions as functions of t

    Parameters
    ----------
    Step_method : func
        The step method that we use to compute numerical results
    Step_name : str
        The name of the algorithm that we use. Used in the legend and title
    dt : float
        The time step
    t_end : float
        The last point of t that we want to compute
    tlim,xlim,ylim : list
        Two element list. Used in the plot to control the limits of both axis. xlim controls the vertical axis of the
        subplot relates to x. Similarly, ylim controls the vertical axis of the subplot relates to y.

    Returns
    --------
    A plot of the numerical results and exact results.
    """
    assert (hasattr(Step_method, '__call__')), 'Step_method must be a callable function'
    assert (isinstance(Step_name, str)), 'Step_name must be a string'
    assert (np.isfinite(dt) and (not np.isnan(dt)) and np.isreal(dt) and \
            (dt >= 0.)), 'dt must be finite, real and positive'
    assert (np.isfinite(t_end) and (not np.isnan(t_end)) and np.isreal(t_end) and \
            (t_end >= 0.)), 'dt must be finite, real and positive'
    assert (len(xlim) == 2) and (len(tlim) == 2) and (len(ylim) == 2), 'The limits of both axis must have length 2'

    stiff_parameters = {'gamma': -2e05, 'omega': 20, 'epsilon': 0.5}
    # Construct the list of t points
    t_list = np.linspace(0., t_end, int(t_end / dt) + 1)  # t_list = np.arange(0.0, t_end + dt, dt)
    # Initialize qn data
    qn = np.array([2 ** 0.5, 3 ** 0.5])
    # Compute numeric results
    qn_list = Runge_Kutta3(Step_method, System_function, 0., qn, dt, t_end, options=stiff_parameters)
    # print(qn_list[:20,:])
    x_numerical = qn_list[:, 0]
    y_numerical = qn_list[:, 1]

    # Exact solution function
    Exact_x = lambda t: np.sqrt(1 + np.cos(t))
    Exact_y = lambda t: np.sqrt(2 + np.cos(stiff_parameters['omega'] * t))

    # Plot required plot
    fig, axes = plt.subplots(1, 2)
    # First subplot
    if Step_name == 'GRRK3':
        axes[0].plot(t_list, x_numerical, 'kx', markersize=4, label='{}'.format(Step_name))
    elif Step_name == 'RK3':
        axes[0].plot(t_list, x_numerical, label='{}'.format(Step_name))
    axes[0].plot(t_list, Exact_x(t_list), label='Exact solution', linewidth=1, color='red')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$x$')
    axes[0].set_xlim(tlim[0], tlim[1])
    axes[0].set_ylim(xlim[0], xlim[1])
    # Second subplot
    if Step_name == 'GRRK3':
        axes[1].plot(t_list, y_numerical,'kx')
    elif Step_name == 'RK3':
        axes[1].plot(t_list, y_numerical)

    axes[1].plot(t_list, Exact_y(t_list), linewidth=1, color='red')
    axes[1].set_xlim(tlim[0], tlim[1])
    axes[1].set_ylim(ylim[0], ylim[1])
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$y$')
    fig.legend()
    fig.suptitle('Task{}: Numerical results of {}'.format(9 if Step_name == 'GRRK3' else 8, Step_name))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Task 10
def Plot_stiff_convergence(t_end=0.05):
    """
    This function is to plot the convergence rate of GRRK3 method applied in stiff problems

    Parameters
    ----------
    t_end : float
        The end point of time t

    Returns
    -------

    """
    assert (np.isfinite(t_end) and (not np.isnan(t_end)) and np.isreal(t_end) and \
            (t_end >= 0.)), 't must be finite, real and positive'

    # Initialize parameters
    stiff_parameters = {'gamma': -2e05, 'omega': 20, 'epsilon': 0.5}
    # Construct the list of dt
    dt_ls = np.array([0.05 / 2 ** j for j in range(8)])
    # Initialize the list of 1-norm errors of both algorithms
    Error_grrk3_ls = np.zeros_like(dt_ls)
    # Exact y solution to the equation
    Exact_y = lambda t: np.sqrt(2 + np.cos(stiff_parameters['omega'] * t))
    # Initial qn data
    qn = np.array([2 ** 0.5, 3 ** 0.5])

    # Loop in list of dt to compute every corresponding 1-norm error
    for i, dt in enumerate(dt_ls):
        # Generate the point of t_ls
        t_ls = np.linspace(0., t_end, int(t_end / dt) + 1)
        # Compute numerical solutions
        y_grrk3_ls = Runge_Kutta3(MyGRRK3_step, System_function, 0, qn, dt, t_end, options=stiff_parameters)[:, 1]
        # Compute the 1-norm error
        # Error_rk3 = dt * np.sum(np.abs(y_grrk3_ls[1:] - Exact_y(t_ls[1:]))) Since the first point is at point x=0
        Error_rk3 = dt * np.sum(np.abs(y_grrk3_ls - Exact_y(t_ls)))
        Error_grrk3_ls[i] = Error_rk3

    # Use polyfit to fit the data
    p_grrk3 = np.polyfit(np.log(dt_ls), np.log(Error_grrk3_ls), 1)

    # Plot required plot
    plt.loglog(dt_ls, Error_grrk3_ls, 'kx', label='Numerical data')
    plt.plot(dt_ls, np.exp(p_grrk3[1]) * dt_ls ** p_grrk3[0], label='Best Fit Line slope{:.3f}'.format(p_grrk3[0]))
    plt.legend()
    plt.xlabel(r'$dt$')
    plt.ylabel(r'$Error$')
    plt.title('Task 10: Convergence rate of GRRK3 for stiff problem')
    plt.show()


def error_vanish(Step_func, dt=0.1):
    """
    This function is to prove that error vanishes at a specific point after a simple step

    Parameters
    ----------
    Step_func ：function
        The step function that used in the unit test

    dt : float
        The step size of t. In this case, dt is also the end of t

    Returns
    -------

    The result of whether the error vanishes after one step

    """
    assert (np.isfinite(dt) and (not np.isnan(dt)) and np.isreal(dt) and \
            (dt >= 0.)), 'dt must be finite, real and positive'

    def f_2(t, q, options=None):
        """
        The RHS of the function defined in question (2)

        Parameters
        ----------

        t : float
            Time point
        q : array of float
            qn
        options : dict
            Other parameters, default set to none

        Returns
        -------

        result_2 : array of float
            2D array. The function value.
        """
        result_2 = np.zeros_like(q)
        result_2[0] = q[1] + 1
        result_2[1] = 6 * t
        return result_2

    q0 = np.array([0, 0])
    # Numerical solution after one step
    truncation = Step_func(f_2, 0, q0, dt)
    # Exact solution
    real = np.array([dt ** 3 + dt, 3 * dt ** 2])
    print(real-truncation)
    assert np.allclose(truncation, real), 'Error still exists'


if __name__ == '__main__':
    start = time.time()
    # Prove that the local truncation error is fourth order
    print('Task 2')
    error_vanish(MyRK3_step)
    print('Task2: Error vanished. Proof completed')
    print('Task 3')
    error_vanish(MyGRRK3_step)
    print('Task5: Error vanished. Proof completed')
    # Plot 1
    Plot_convergence_rate(MyRK3_step, 'RK3')
    plt.figure()
    # Plot 2
    Plot_convergence_rate(MyGRRK3_step, 'GRRK3')
    # Plot 3
    Plot_nonstiff_rk3_grrk3()
    # Plot 4
    Plot_nonstiff_convergence()
    # Plot 5
    # Plot shows the failure of the RK3 method
    Plot_stiff_solution(MyRK3_step, 'RK3', 0.001, t_end=0.02, tlim=[0, 0.02], xlim=[1, 2], ylim=[1, 2.5])
    # Plot 6
    Plot_stiff_solution(MyGRRK3_step, 'GRRK3', 0.005, t_end=1, tlim=[0., 1.], xlim=[1.24, 1.42], ylim=[1.0, 1.8])
    plt.figure()
    # Plot 7
    Plot_stiff_convergence()
    stop = time.time()
    print('Running time is {:.2f}s'.format(stop - start))
