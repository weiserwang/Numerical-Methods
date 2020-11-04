import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy import optimize
from scipy import linalg

"""
You should change the following variable to match your student ID
"""
student_id = 31431127

"""
This next lines should not be modified
"""
np.random.seed(student_id)
A = np.random.randint(1, 10, size=(3, 3))
while np.linalg.det(A) < 1 and np.linalg.cond(A) > 1e4:
    A = np.random.randint(1, 10, size=(3, 3))
b = np.random.randint(1, 10, size=(3,))

"""
You should modify each function so that the required solution to question_N
is *returned* from the function question_N. So, running question_N() in the
console should display the correct answer to the screen.

When plots are required they should be displayed using plt.show(), and nothing
returned by the function.
"""


def question_1():
    """
    Solution to question 1 goes here
    """
    M = np.array([[1, 3], [4, 5]])
    return M.T


def question_2():
    """
    Solution to question 2 goes here
    """
    v = np.linspace(1, 2, 40)
    return np.dot(v, v)


def question_3():
    """
    Solution to question 3 goes here
    """
    # Create vector v
    v = np.linspace(1, 2, 40)
    # Create vector w
    w = v ** 2 * np.cos(np.pi * v)
    return np.sum(w[0::2])


def question_4():
    """
    Solution to question 4 goes here
    """
    z = np.linalg.solve(A, b)
    return z


def question_5():
    """
    Solution to question 5 goes here
    """
    # Calculate the eigenvalues of the matrix
    eigenvalues, _ = np.linalg.eig(A - 2 * np.eye(len(A)))
    # Find the smallest and largest magnitude eigenvalues
    lambda1 = np.min(np.abs(eigenvalues))
    lambda3 = np.max(np.abs(eigenvalues))
    return lambda1, lambda3


def question_6():
    """
    Solution to question 6 goes here
    """
    F = np.array([[1, 2], [3, 4]])
    G = scipy.linalg.expm(F)
    H = np.exp(F)
    return G, H


def question_7():
    """
    Solution to question 7 goes here
    """
    x = np.linspace(1, 4, 300)
    # The function y(x)
    y = lambda x: np.log(x) * np.sin(2 * np.pi * x)
    # Plot the function
    plt.plot(x, y(x))
    plt.xlabel('x')
    plt.ylabel(r'y = $\log(x)\sin(2 \pi x)$')
    plt.title('Figure for question 7')
    plt.show()


def question_8():
    """
    Solution to question 8 goes here
    """
    t = 2 ** (-1 * np.arange(11, dtype=np.float))
    e = np.abs(np.sin(t) - t * np.cos(t))
    # Plot the figure
    plt.loglog(t, e, 'kx')
    plt.xlabel('t')
    plt.ylabel('e')
    plt.show()


def question_9():
    """
    Solution to question 9 goes here
    """

    def func(x):
        """
        The function to be integrated
        """
        return np.exp(-1 * x ** 2) * np.cos(x) / (1 + x ** 3)

    # Use scipy function to calculate the quadrature
    I = scipy.integrate.quad(func, 0, 2)[0]
    return I


def question_10():
    """
    Solution to question 10 goes here
    """

    def yn(u0, v0, seq_len):
        """
        Parameters
        ----------
        u0,v0 : float
            The initial data of the vector sequence
        seq_len : int
            The total number of yn points which exclude y0

        Returns
        -------
        y_list : list
            The list of calculated y points and the length should be seq_len+1
        """
        y_list = [[u0, v0]]  # Initialize the list of y
        for i in range(1, seq_len + 1):
            # New points
            u_new, v_new = np.cos(y_list[i - 1][1]), np.sin(y_list[i - 1][0])
            # Add new points to the y_list
            y_list.append([u_new, v_new])
        return y_list

    y_list = np.array(yn(0.25, 0.75, 50), dtype=float)
    return y_list[5], y_list[50]
