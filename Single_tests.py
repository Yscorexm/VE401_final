from scipy import stats
import numpy as np
from collections import Counter

def mean_test(mean, s, n, delta=0, H0=None, variance="known", CI=False, alpha=0.05):
    '''
    Z-test or T-test for the mean.  
    H0: μ ≤, ≥, = delta.  
    In slides 349, .  
    REQUIRE: H0 can take three values: "equal", "less", "greater".  
    variance: "unknown", "known". 
    if variance == "known", s will be treated as σ.   
    Z for "known" and "large", T for "unknown".  
    CI: True, False.  
    RETURN: CI False: Test statistics, critical value, p-value.  
    CI True: CI for μ.
    '''
    if CI:
        delta = 0
    nominator = mean - delta
    denominator = s / n**0.5
    statistic = nominator / denominator
    if variance == "known":
        distribution = stats.norm
    elif variance == "unknown":
        distribution = stats.t(n-1)
    
    if H0 == "less":
        c_value = distribution.ppf(1-alpha)
        p_value = 1 - distribution.cdf(statistic)
    elif H0 == "greater":
        c_value = distribution.ppf(alpha)
        p_value = distribution.cdf(statistic)
    elif H0 == "equal":
        c_value = distribution.ppf(1-alpha/2)
        p_value = 2 * min(1 - distribution.cdf(statistic), distribution.cdf(statistic))
    if not CI:
        return statistic, c_value, p_value
    else:
        diff = distribution.ppf(1-alpha/2)*denominator
        return nominator - diff, nominator + diff


def chi2_test(s, n, sigma_0=0, H0=None, CI=False, alpha=0.05):
    '''
    Chi-Squared Test for the Variance.  
    H0: σ ≤, ≥, = σ0.  
    In slides 408.  
    REQUIRE: H0 can take three values: "equal", "less", "greater".  
    CI: True, False.  
    RETURN: CI False: Test statistics, critical value, p-value.  
    CI True: CI for μ.
    '''
    if CI:
        sigma_0 = 0
    X2 = (n-1)*s**2 / sigma_0**2
    distribution = stats.chi2(n-1)
    if H0 == "less":
        c_value = distribution.ppf(1-alpha)
        p_value = 1 - distribution.cdf(X2)
    elif H0 == "greater":
        c_value = distribution.ppf(alpha)
        p_value = distribution.cdf(X2)
    elif H0 == "equal":
        c_value = (distribution.ppf(alpha/2), distribution.ppf(1-alpha/2))
        p_value = 2 * min(1 - distribution.cdf(X2), distribution.cdf(X2))
    if not CI:
        return X2, c_value, p_value
    else:
        return distribution.ppf(alpha/2)*sigma_0**2/(n-1), X2 + distribution.ppf(1-alpha/2)*sigma_0**2/(n-1)
    



def sign_test(X, H0):
    '''
    Sign Test for the Median.  
    H0: M(X) ≤, ≥, = 0.  
    In slides 418-420.  
    REQUIRE: H0 can take three value: "equal", "less", "greater".  
    RETURN: (Q_minus, Q_plus), p-value.  
    IMPORTANT: to be tested.  
    '''
    X1 = [bool(k < 0) for k in X if k != 0]
    n = len(X1)
    Q_minus = sum(X1)
    Q_plus = n - Q_minus
    distribution = stats.binom(n, 0.5)
    if H0 == "less":
        p_value = distribution.cdf(Q_minus)
    elif H0 == "greater":
        p_value = distribution.cdf(Q_plus)
    elif H0 == "equal":
        p_value = 2 * distribution.cdf(min(Q_minus, Q_plus))
    return (Q_minus, Q_plus), p_value


def wsr_test(X, H0):
    '''
    Wilcoxon Signed Rank Test.  
    H0: M(X) ≤, ≥, = 0.  
    In slides 427.  
    REQUIRE: H0 can take three value: "equal", "less", "greater".  
    RETURN: (E[w], Var[w]), (w_minus, w_plus), p-value.     
    '''
    d = np.asarray([k for k in X if k != 0])
    n = len(d)
    if n < 10:
        print("Sample size too small for normal approximation.")
    r = stats.rankdata(np.abs(d))
    # print(r)
    w_plus = np.sum((d > 0) * r, axis=0)
    w_minus = np.sum((d < 0) * r, axis=0)
    E_w = n*(n+1)/4
    Var_w = n*(n+1)*(2*n+1)/24

    replist, repnum = stats.find_repeats(r)
    print(repnum)
    if repnum.size != 0:
        # Correction for repeated elements.
        Var_w -= (repnum * (repnum * repnum - 1)).sum()/48
    
    if H0 == "less":
        Z = (w_minus - E_w) / Var_w**0.5
        p_value = stats.norm.cdf(Z)
    elif H0 == "greater":
        Z = (w_minus - E_w) / Var_w**0.5
        p_value = stats.norm.cdf(Z)
    elif H0 == "equal":
        Z = (min(w_minus, w_plus) - E_w) / Var_w**0.5
        p_value = 2 * stats.norm.cdf(Z)
    return (E_w, Var_w), (-w_minus, w_plus), p_value




def proportion2_test(p, n, p0, H0, alpha=0.05):
    '''
    Test for Comparing Two Proportions.  
    H0: p ≤, ≥, = p0.  
    In slides 443.  
    REQUIRE: H0 can take three value: "equal", "less", "greater".  
    RETURN: Test statistics, critical value, p-value.  
    IMPORTANT: to be tested.
    '''
    Z = (p - p0) / (p0 * (1-p0) / n)**0.5
    if H0 == "less":
        z = stats.norm.ppf(1-alpha)
        p_value = 1 - stats.norm.cdf(Z)
    elif H0 == "greater":
        z = stats.norm.ppf(alpha)
        p_value = stats.norm.cdf(Z)
    elif H0 == "equal":
        z = stats.norm.ppf(1-alpha/2)
        p_value = 2 * min(1 - stats.norm.cdf(Z), stats.norm.cdf(Z))
    return Z, z, p_value


# 14.4 
# print(mean_test(41.25, 2, 25, 40, H0="equal", variance="known"))

# 17.4
print(chi2_test(1.41, 20, 1.5, H0="greater"))

# 18.3 18.5
# X = [5, 1, 5, 4, 4, 6, 6,
#      3, 6, 2, 3, 5, 5, 6,
#      4, 4, 4, 3, 3, 4]
# X1 = [k-3.5 for k in X]
# print(sign_test(X1, "less"))
# print(stats.wilcoxon(X1, alternative='greater'))
# print(wsr_test(X1, "less"))


'''
to do:
slides 461: Neyman-Pearson test of Z-test, chi2-test, T-test.
slides 463-486: Neyman-Pearson test of comparison of means.
'''