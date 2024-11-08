'''
Functions for Exercise 9
'''
# Linear regression
def linregress(x, y):
    """
    Function that calculates the slopes and y-intercepts for unweighted linear regression lines.

    Parameters
    ----------
    x
    y

    Returns
    -------
    A: y-intercept
    B: slope
    
    """
    delta = (len(x)*(x**2).sum()) - ((x.sum())**2)
    A = (((x**2).sum() * y.sum()) - (x.sum() * (x*y).sum()))/delta
    B = ((len(x)*((x*y).sum())) - (x.sum()*y.sum()))/delta
    return A,B

# Pearson Coefficient
import numpy as np
def pearson(x, y):
    """
    Function that calculates the correlation coefficient

    Parameters
    ----------
    x
    y

    Returns
    -------
    r: correlation coefficient
    
    """
    
    # initialize summing variables...
    
    topsum = 0
    bottomsumx = 0
    bottomsumy = 0

    for i in range(len(x)):
        topsum = topsum + (x[i] - x.mean()) * (y[i] - y.mean())
        bottomsumx = bottomsumx + (x[i] - x.mean())**2
        bottomsumy = bottomsumy + (y[i] - y.mean())**2

        # actually calculate r
        r = topsum / (np.sqrt(bottomsumx * bottomsumy)) 
    return r

# Goodness-of-fit value
def chi_squared(obs, exp, std):
    '''
    Function to calculate the ggodness-of-fit

    Parameters
    ----------
    obs: observed value
    exp: expected value
    std: standard deviation

    Returns
    -------
    cs: the reduced chi-squared value
    '''
    inside = 0
    
    for i in range(len(obs)):
        inside += ((obs[i] - exp[i])**2)/((std[i])**2)
    cs = (1/len(obs))*inside
    return cs

