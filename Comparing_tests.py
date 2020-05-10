from scipy import stats
import math


def proportion2_test(p1, p2, n1, n2, d, H0, alpha=0.05):
    '''
    Test for Comparing Two Proportions.  
    H0: p1-p2 ≤, ≥, = d.  
    In slides 446.  
    REQUIRE: H0 can take three value: "equal", "less", "greater".  
    RETURN: Test statistics, critical value, p-value.  
    IMPORTANT: to be tested.
    '''
    Z = (p1-p2-d)/(p1*(1-p1)/n1 + p2*(1-p2)/n2)**0.5
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


def pooled_proportion2_test(p1, p2, n1, n2, H0, alpha=0.05):
    '''
    Pooled Test for Equality of Proportions.  
    H0: p1 ≤, ≥, = p2.  
    In slides 448.  
    REQUIRE: H0 can take three value: "equal", "less", "greater".  
    RETURN: Test statistics, critical value, p-value.  
    '''
    pooled_p = (n1*p1 + n2*p2) / (n1+n2)
    Z = (p1-p2)/(pooled_p * (1-pooled_p) * (1/n1 + 1/n2))**0.5
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


def F_test(s1, s2, n1, n2, H0, alpha=0.05):
    '''
    F-Test for comparison of two variances.  
    H0: σ1 ≤, ≥, = σ2.  
    In slides 458.  
    REQUIRE: H0 can take three values: "equal", "less", "greater".  
    RETURN: Test statistics, critical value, p-value.  
    '''
    F = s1**2 / s2**2
    if H0 == "less":
        c_value = stats.f(n1-1, n2-1).ppf(1-alpha)
        p_value = 1 - stats.f(n1-1, n2-1).cdf(F)
    elif H0 == "greater":
        c_value = stats.f(n1-1, n2-1).ppf(alpha)
        p_value = stats.f(n1-1, n2-1).cdf(F)
    elif H0 == "equal":
        F1, F2 = F, 1/F
        c_value1, c_value2 = stats.f(n1-1, n2-1).ppf(1-alpha/2), stats.f(n2-1, n1-1).ppf(1-alpha/2)
        F = (F1, F2)
        c_value = (c_value1, c_value2)
        p_value = 2 * min(1 - stats.f(n1-1, n2-1).cdf(F1), 1 - stats.f(n2-1, n1-1).cdf(F2))
    return F, c_value, p_value



def mean2_test(mean1, mean2, s1, s2, n1, n2, delta=0, H0=None, variance="known", sample_size="small", CI=False, alpha=0.05):
    '''
    Z-test or T-test for comparing means.  
    H0: μ1 - μ2 ≤, ≥, = d.  
    In slides 463-486.  
    REQUIRE: H0 can take three values: "equal", "less", "greater".  
    variance: "unknown", "known", "unknown_equal", sample_size: "small", "large".  
    if variance == "known", s1 and s2 will be treated as σ1 and σ2.  
    Z for "unknown" and "large", Tn1+n2-2 for "unknown_equal", Tγ for "unknown" and "small".  
    CI: True, False.  
    RETURN: CI False: Test statistics, critical value, p-value.  
    CI True: CI for μ1 - μ2.
    '''
    if CI:
        delta = 0
    nominator = mean1 - mean2 - delta
    if variance == "known":
        denominator = (s1**2 / n1 + s2**2 / n2)**0.5
    elif variance == "unknown" and sample_size == "large":
        denominator = (s1**2 / n1 + s2**2 / n2)**0.5
    elif variance == "unknown_equal":
        df = gamma = n1 + n2 - 2
        Sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
        print("Sp_square = ", Sp2)
        denominator = (Sp2 * (1/n1 + 1/n2))**0.5
    elif variance == "unknown" and sample_size == "small":
        df = gamma = math.floor((s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2 / (n1-1) + (s2**2/n2)**2 / (n2-1)))
        print("γ = ", gamma)
        denominator = (s1**2 / n1 + s2**2 / n2)**0.5
    
    statistic = nominator / denominator
    if variance == "known" or (variance == "unknown" and sample_size == "large"):
        distribution = stats.norm
    else:
        distribution = stats.t(df) # distribution

    if H0 == "less":
        c_value = distribution.ppf(1-alpha)
        p_value = 1 - distribution.cdf(statistic)
    elif H0 == "greater":
        c_value = distribution.ppf(alpha)
        p_value = distribution.cdf(statistic)
    elif H0 == "equal":
        c_value = (distribution.ppf(alpha/2), distribution.ppf(1-alpha/2))
        p_value = 2 * min(1 - distribution.cdf(statistic), distribution.cdf(statistic))
    if not CI:
        return statistic, c_value, p_value
    else:
        diff = distribution.ppf(1-alpha/2)*denominator
        return mean1 - mean2 - diff, mean1 - mean2 + diff

    def residue_plot(self):
        e = [self.y[i] - self.predict(self.x[i]) for i in range(self.n)]
        fig=plot.figure(num=1, figsize=(15, 8),dpi=80)
        plot.scatter(self.x,e)
        plot.show()

def get_mean_variance(X):
    '''
    RETURN: mean, s, n
    '''
    n = len(X)
    mean = sum(X) / n
    s = (sum([(k-mean)**2 for k in X]) / (n-1))**0.5
    return mean, s, n



# 19.6
# print(pooled_proportion2_test(0.08, 0.06, 100, 200, "less"))

# 20.7
# print(F_test(4.02**0.5, 3.89**0.5, 8, 8, "equal", 0.2))

# 21.1
# print(mean2_test(824.9, 818.6, 40**0.5, 50**0.5, 10, 10, delta=0, H0="less", variance="known"))
# 21.4
# print(mean2_test(24.6, 22.1, 0.85, 0.98, 12, 15, variance="unknown_equal", sample_size="small", CI=True))
# 21.6
# print(mean2_test(0.9375, 0.9173, 0.0389, 0.0402, 8, 8, delta=0, H0="less", variance="unknown_equal", sample_size="small"))


# X = [75, 46, 57, 43, 58, 38, 61, 56, 64, 65]
# Y = [52, 41, 43, 47, 32, 49, 52, 44, 57, 60]
# print(get_mean_variance(X))
# print(get_mean_variance(Y))
# print(mean2_test(56.3, 47.7, 11.20, 8.19, 10, 10, 0, H0="equal", variance="unknown", sample_size="small"))

print(mean2_test(121, 112, 8, 8, 10, 10, 0, H0="less", variance="known"))

'''
to do:
slides 461: Neyman-Pearson test of F-test
slides 463-486: Neyman-Pearson test of comparison of means.
'''