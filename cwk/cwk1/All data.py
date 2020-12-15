import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from RK3_Method import MyRK3_step
from RK3_Method import MyGRRK3_step
from RK3_Method import Runge_Kutta3
import time


# Question 2
def q2_error_vanish(t_end=0.1):
    """
    This function is to prove that error vanishes at a specific point after a simple step

    Parameters
    ----------
    t_end:

    Returns
    -------
    The result of whether the error vanishes after one step
    """

    def f_2(t, q, options=None):
        """
        The RHS of the function defined in question 2

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
        result_2 : array of float
            2D array. The function value

        """
        result_2 = np.zeros_like(q)
        result_2[0] = q[1] + 1
        result_2[1] = 6 * t
        return result_2

    q_2 = np.array([0, 0])
    dt_2 = t_end
    truncation = MyRK3_step(f_2, 0, q_2, dt_2)
    real = np.array([t_end ** 3 + t_end, 3 * t_end ** 2])
    if np.allclose(truncation, real) is True:
        return 'Error vanishes. Proof completed'
    else:
        return 'Error Still Exist'


def Plot_convergence_rate(step_func, method_name, t_end=0.1):
    """
    This function is to plot the convergence rate of explicit RK3 method

    Parameters
    ----------
    step_func : function
        The step method used to check the convergence rate
    method_name : str
        The name of the algorithm. Used in title of the plot

    t_end: float
        The point that we use to compute error. Default to 0.6


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

    # Generate a list of N
    Npoints = np.array([2 ** i for i in range(10)])
    # Determine the step size
    dt_3 = t_end / Npoints
    # Initialize the list of error
    error_3 = np.zeros_like(dt_3)
    # Initialize the point
    qn_3 = np.array([0], dtype=float)
    # The ture result of qn at point x_end
    real_result = np.exp(-1 * t_end) + t_end - 1
    # Compute the error
    for i in range(len(Npoints)):
        truncated_result = Runge_Kutta3(step_func, f3, 0, qn_3, dt_3[i], Npoints[i])[-1]  # Fetch last result
        error_3[i] = np.abs(truncated_result - real_result)
    # Fit the points with a straight line
    p_3 = np.polyfit(np.log(dt_3), np.log(error_3), 1)
    # Plot the required convergence rate
    plt.loglog(dt_3, error_3, 'kx', label='Numerical Data')
    plt.loglog(dt_3, np.exp(p_3[1]) * dt_3 ** p_3[0], "b-", label='Line slope {:.2f}'.format(p_3[0]))
    plt.xlabel(r'$dt$')
    plt.ylabel(r'$|Error|$')
    plt.title('Convergence Rate for {} method'.format(method_name))
    plt.legend()
    plt.tight_layout()
    plt.show()


# Question 6
def System_function(t, q, options):
    """
    This the function which defined in coursework 1 equation 2

    Parameters
    ----------
    t : float
        The time
    q : array of float
        qn data
    options : dict
        The dictionary containing other necessary parameters

    """
    # Fetch parameters from options argument
    gamma = options.get('gamma')
    w = options.get('w')
    epsilon = options.get('epsilon')
    # Construct parameter matrix
    parameter_matrix = np.array([[gamma, epsilon], [epsilon, -1]])
    # Construct the left matrix in the function
    x, y = q[0], q[1]
    middle_matrix = np.array([[(-1 + x ** 2 - np.cos(t)) / (2 * x)],
                              [(-2 + y ** 2 - np.cos(w * t)) / (2 * y)]])
    right_matrix = np.array([[np.sin(t) / (2 * x)],
                             [w * np.sin(w * t) / (2 * y)]])
    result_sys = np.dot(parameter_matrix, middle_matrix) - right_matrix
    result_sys = result_sys.reshape(-1, )  # Guarantee the shape of the returned vector
    return result_sys


def Plot_nonstiff_rk3_grrk3():
    """
    This function is the required plot of question 6.

    Returns
    -------
    A figure which shows the numerical of two algorithms and the exact result
    """
    # The parameters in equation 2
    q6_parameters = {'gamma': -2, 'w': 5, 'epsilon': 0.05}
    # dt
    dt_6 = 0.05
    # Compute evenly spaced points
    t_list = np.arange(0.0, 1.0 + dt_6, dt_6)
    # Initial qn data
    qn = np.array([2 ** 0.5, 3 ** 0.5])
    # print(System_function(0, qn, q6_parameters))
    # x_rk3_list = np.zeros_like(t_list)
    # x_rk3_list[0] = qn[0]
    # y_rk3_list = np.zeros_like(t_list)
    # y_rk3_list[0] = qn[1]
    # x_grrk3_list = np.zeros_like(t_list)
    # x_grrk3_list[0] = qn[0]
    # y_grrk3_list = np.zeros_like(t_list)
    # y_grrk3_list[0] = qn[1]
    # # Initialize qn for rk3 method and grrk3 method respectively
    # qn_rk3 = qn.copy()
    # qn_grrk3 = qn.copy()
    # qn_test = qn.copy()
    # # Derive numerical solutions
    # for i, t_point in enumerate(t_list[:-1]):
    #     # RK3 method
    #     qn_rk3 = MyRK3_step(System_function, t_point, qn_rk3, dt_6, options=q6_parameters)
    #     x_rk3_list[i + 1] = qn_rk3[0]
    #     y_rk3_list[i + 1] = qn_rk3[1]
    #     # GRRK3 method
    #     qn_grrk3 = MyGRRK3_step(System_function, t_point, qn_grrk3, dt_6, options=q6_parameters)
    #     x_grrk3_list[i + 1] = qn_grrk3[0]
    #     y_grrk3_list[i + 1] = qn_grrk3[1]

    x_rk3_list = Runge_Kutta3(MyRK3_step, System_function, 0, qn, dt_6, 1 / dt_6, options=q6_parameters)[:, 0]
    y_rk3_list = Runge_Kutta3(MyRK3_step, System_function, 0, qn, dt_6, 1 / dt_6, options=q6_parameters)[:, 1]
    x_grrk3_list = Runge_Kutta3(MyGRRK3_step, System_function, 0, qn, dt_6, 1 / dt_6, options=q6_parameters)[:, 0]
    y_grrk3_list = Runge_Kutta3(MyGRRK3_step, System_function, 0, qn, dt_6, 1 / dt_6, options=q6_parameters)[:, 1]

    # The exact solution of the function
    exact_x = lambda t: np.sqrt(1 + np.cos(t))
    exact_y = lambda t: np.sqrt(2 + np.cos(q6_parameters['w'] * t))
    # Initialize two subplots
    fig, axes = plt.subplots(1, 2)
    # LHS plot
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1.24, 1.42)
    axes[0].plot(t_list, x_rk3_list, label='RK3', linewidth=6)
    axes[0].plot(t_list, x_grrk3_list, label='GRRK3', linewidth=3)
    axes[0].plot(t_list, exact_x(t_list), label='Exact Solution', linewidth=1)
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$x$')
    # RHS plot
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 1.8)
    axes[1].plot(t_list, y_rk3_list, linewidth=6)
    axes[1].plot(t_list, y_grrk3_list, linewidth=3)
    axes[1].plot(t_list, exact_y(t_list), linewidth=1)
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$y$')
    fig.legend()
    fig.tight_layout()
    plt.show()


# Question 7
q7_parameters = {'gamma': -2, 'w': 5, 'epsilon': 0.05}
dt_7 = np.array([0.1 / 2 ** j for j in range(8)])


# Question 8/9
def Plot_stiff_solution(Step_method, Step_name, dt):
    q8_parameters = {'gamma': -2 * 10 ** 5, 'w': 20, 'epsilon': 0.5}
    # Construct t_list
    t_list = np.arange(0.0, 1.0 + dt, dt)
    # Initialize qn data
    qn = np.array([2 ** 0.5, 3 ** 0.5])
    # Initialize the list of x and y
    # x_q8 = np.zeros_like(t_list)
    # x_q8[0] = qn[0]
    # y_q8 = np.zeros_like(t_list)
    # y_q8[0] = qn[1]
    # # Compute qn
    # for i, t_point in enumerate(t_list[:-1]):
    #     qn = Step_method(System_function, t_point, qn, dt, options=q8_parameters)
    #     x_q8[i + 1] = qn[0]
    #     y_q8[i + 1] = qn[1]
    # Compute the list of qn
    qn_list = Runge_Kutta3(Step_method, System_function, 0., qn, dt, 1 / dt, options=q8_parameters)
    x_numerical = qn_list[:, 0]
    y_numerical = qn_list[:, 1]

    # Exact solution function
    Exact_x = lambda t: np.sqrt(1 + np.cos(t))
    Exact_y = lambda t: np.sqrt(2 + np.cos(q8_parameters['w'] * t))

    # Plot required plot
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(t_list, x_numerical, label='{}'.format(Step_name), linewidth=3)
    axes[0].plot(t_list, Exact_x(t_list), label='Exact solution', linewidth=1, color='red')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$x$')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-1, 1.8)
    axes[1].plot(t_list, y_numerical, linewidth=5)
    axes[1].plot(t_list, Exact_y(t_list), linewidth=1, color='red')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$y$')
    fig.legend()
    fig.canvas.set_window_title('None-Stiff')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    start = time.time()
    # Plot 1
    Plot_convergence_rate(MyRK3_step, 'RK3')
    plt.figure()
    # Plot 2
    Plot_convergence_rate(MyGRRK3_step, 'GRRK3')
    #Plot 3
    Plot_nonstiff_rk3_grrk3()
    #Plot 5
    Plot_stiff_solution(MyRK3_step, 'RK3', 0.001)
    # Plot 6
    Plot_stiff_solution(MyGRRK3_step, 'GRRK3', 0.005)
    stop = time.time()
    print(stop-start)
