from scipy import stats
from Single_tests import wsr_test
import math

def paired_T_test(X, Y, H0, alpha=0.05):
    '''
    Paired T-Test.  
    H0: mean_D = or <= or >= 0.  
    In slides 502.  
    REQUIRE: H0 can take three values: "less", "greater", "equal". 
    RETURN: Test statistics, critical value, p-value.
    '''
    n = len(X)
    D = [X[i]-Y[i] for i in range(len(X))]
    mean_D = sum(D) / n
    S_D = (sum([(k-mean_D)**2 for k in D]) / (n-1))**0.5
    T = mean_D * n**0.5 / S_D
    if H0 == "less":
        c_value = stats.t(n-1).ppf(1-alpha)
        p_value = 1 - stats.t(n-1).cdf(T)
    elif H0 == "greater":
        c_value = stats.t(n-1).ppf(alpha)
        p_value = stats.t(n-1).cdf(T)
    else:
        c_value = stats.t(n-1).ppf(1-alpha)
        p_value = 2 * min(1 - stats.t(n-1).cdf(T), stats.t(n-1).cdf(T))
    return T, c_value, p_value


def paired_wsr_test(X, Y, delta):
    '''
    Paired Wilcoxon signed rank test.
    H0: P[X − Y > δ] = P[X′ − Y > 0] = 1/2 
    In slides 507.   
    RETURN: (E[w], Var[w]), (w_minus, w_plus), p-value.     
    '''
    D = [X[i]-Y[i] for i in range(len(X))]
    return wsr_test(D, "greater")
    

def CI_correlation(X, Y, alpha=0.05):
    '''
    Confidence Interval for the Correlation Coefficient.  
    In slides 519.  
    RETURN: CI.  
    IMPORTANT: to be tested.
    '''
    n = len(X)
    X_mean = sum(X) / n
    Y_mean = sum(Y) / n
    Sxx = sum([(k - X_mean)**2 for k in X])
    Syy = sum([(k - Y_mean)**2 for k in Y])
    Sxy = sum([(X[i] - X_mean) * (Y[i] - Y_mean) for i in range(n)])
    R = Sxy / (Sxx * Syy)**0.5
    z = stats.norm.ppf(1-alpha/2)
    lower = math.tanh(math.atanh(R) - z / (n-3)**0.5)
    upper = math.tanh(math.atanh(R) + z / (n-3)**0.5)
    return lower, upper


# 22.4
X = [209, 178, 169, 212, 180, 192, 158, 180,
     170, 153, 183, 165, 201, 179, 243, 144]
Y = [196, 171, 170, 207, 177, 190, 159, 180,
     164, 152, 179, 162, 199, 173, 231, 140]
print(paired_T_test(X, Y, "less", 0.01))
