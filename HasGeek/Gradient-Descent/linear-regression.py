import numpy as np

def gradient_descent(x, y):
    #initial value of m and b
    m_curr = b_curr = 0
    #initialize number of steps
    iterations = 1600
    #Number of data points n
    n = len(x)
    #Initialize learning rate
    learning_rate = 0.01

    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
    

    print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x, y)