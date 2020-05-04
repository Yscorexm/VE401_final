import numpy as np 
from scipy import stats

def GNoF_test(O, E, m=0, alpha=0.05):
    '''
    Pearson's Chi-squared Goodness-of-Fit Test.  
    H0: pi = pi_0.  
    In slides 533 and 536.  
    REQUIRE: O and E are lists of the same size k.  
             m is the number of estimator.  
    RETURN: Test statistics, critical value, p-value.
    '''
    k = len(O)
    X_square = sum([(O[i] - E[i])**2 / E[i] for i in range(k)])
    chi_square = stats.chi2(k - 1 - m).ppf(1-alpha)
    p_value = 1 - stats.chi2(k - 1 - m).cdf(X_square)
    return X_square, chi_square, p_value


def indepence_test(a, alpha=0.05):
    '''
    Chi-Squared Test for Independence.  
    H0: Row and column categorizations are independent.
    In slides 549-552.  
    REQUIRE: a is numpy 2-D array.  
    RETURN: Test statistics, critical value, p-value.
    '''
    r, c = a.shape
    n = np.sum(a)
    n0 = np.sum(a, axis=0)
    n1 = np.sum(a, axis=1)
    E = np.zeros_like(a)
    for i in range(r):
        for j in range(c):
            E[i, j] = n0[j] * n1[i] / n
    
    X_square = np.sum((a - E)**2 / E)
    df = (r-1)*(c-1) # degrees of freedom
    critical_value = stats.chi2(df).ppf(1-alpha)
    p_value = 1 - stats.chi2(df).cdf(X_square)
    return X_square, critical_value, p_value


# 23.10
# O = [94, 93, 112, 101, 104, 95, 100, 99, 94, 108]
# E = [100] * 10
# print(GNoF_test(O, E, 0))


# 23.14
# a = np.array([[160, 140, 40], [40, 60, 60]])
# print(indepence_test(a))
