import numpy as np
from scipy import stats


class SLR:
    '''
    simple linear regression
    y = b0 + b1x + e
    '''

    def __init__(self, n, x, y):
        self.n = n
        self.x = x
        self.y = y
        self.x_mean = sum(x) / n
        self.y_mean = sum(y) / n
        self.sum_x2 = sum([k**2 for k in x])
        self.sum_y2 = sum([k**2 for k in y])
        self.Sxx = sum([(k - self.x_mean)**2 for k in x])
        self.Syy = sum([(k - self.y_mean)**2 for k in y])
        self.Sxy = sum([(x[i] - self.x_mean) * (y[i] - self.y_mean) for i in range(n)])
        self.SST = self.Syy
        self.b1 = self.Sxy / self.Sxx
        self.b0 = self.y_mean - self.b1 * self.x_mean
        self.SSE = self.Syy - self.b1 * self.Sxy
        self.SSR = self.SST - self.SSE
        self.R_square = self.SSR / self.SST
        self.parameters = (self.b0, self.b1)
        self.S = (self.SSE / (n-2))**0.5
        self.e = [y[i] - self.predict(x[i]) for i in range(n)]

    def predict(self, x_0):
        '''
        RETURN: Estimated Mean at x = x_0.
        '''
        return self.b1 * x_0 + self.b0

    def CI_parameters(self, alpha=0.05):
        '''
        Confidence Intervals on the Slope and the Intercept.  
        In slides 580.  
        RETURN: CI tuple, first for β0 and second for β1.
        '''
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        diff_1 = t * self.S / self.Sxx**0.5
        diff_0 = t * self.S * (self.sum_x2 / (self.n * self.Sxx))**0.5
        CI_beta0 = (self.b0 - diff_0, self.b0 + diff_0)
        CI_beta1 = (self.b1 - diff_1, self.b1 + diff_1)
        return CI_beta0, CI_beta1

    def significance_test(self, alpha=0.05):
        '''
        T-Test for Significance of Regression.  
        H0: β1 = 0.
        In slides 585.  
        RETURN: Test statistics and critical value.  
        '''
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        T = self.b1 * self.Sxx**0.5 / self.S
        return T, t

    def CI(self, x_0, alpha=0.05):
        '''
        Confidence Interval for the Estimated Mean at x = x_0.  
        In slides 589.  
        REQUIRE: x_0.  
        RETURN: CI.  
        IMPORTANT: to be tested.
        '''
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        mean = self.predict(x_0)
        diff = t * self.S * (1/self.n + (x_0 - self.x_mean)**2/self.Sxx)**0.5
        print(mean, diff)
        return mean-diff, mean+diff
    
    def PI(self, x_0, alpha=0.05):
        '''
        Prediction Interval for the observed value at x = x_0.  
        In slides 593.  
        REQUIRE: x_0.  
        RETURN: PI.
        IMPORTANT: to be tested.
        '''
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        mean = self.predict(x_0)
        diff = t * self.S * (1 + 1/self.n + (x_0 - self.x_mean)**2/self.Sxx)**0.5
        print(mean, diff)
        return mean-diff, mean+diff
    
    def correlation_test(self, alpha=0.05):
        '''
        Test for Correlation.  
        H0: ϱ = 0.  
        In slides 601.  
        RETURN: Test statistics and critical value.  
        IMPORTANT: to be tested.
        '''
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        T = (self.R_square * (self.n - 2) / (1 - self.R_square))**0.5
        return T, t
    

def LoF_test(N, X, Y, alpha=0.05):
    '''
    Test for Lack of Fit.  
    H0: the linear regression model is appropriate.  
    In slides 608.  
    REQUIRE: multiple sampling for single x. N, X are 1-D lists. Y is a 2-D list.  
    RETURN: (SSE, SSE_pe, SSE_if), (Test statistics and critical value).
    '''
    k = len(N)
    n = sum(N)
    print(n, k)
    mean_Y = [sum(Y[i])/N[i] for i in range(k)]
    SSE_pe = sum(sum([(Y[i][j] - mean_Y[i])**2 for j in range(N[i])]) for i in range(k))
    x = []
    for i in range(k):
        x.extend([X[i]] * N[i])
    y = []
    for i in range(k):
        y.extend(Y[i])
    model = SLR(n, x, y)
    SSE = model.SSE
    SSE_if = SSE - SSE_pe
    F = (SSE_if / (k - 2)) / (SSE_pe / (n - k))
    f = stats.f(k - 2, n - k).ppf(1-alpha)
    return (SSE, SSE_pe, SSE_if), (F, f)






# 24.1 24.5 24.7
# x = [35.3, 27.7, 30.8, 58.8, 61.4, 71.3, 74.4, 76.7, 70.7, 57.5,
#      46.4, 28.9, 28.1, 39.1, 46.8, 48.5, 59.3, 70.0, 70.0, 74.4,
#      72.1, 58.1, 44.6, 33.4, 28.6]
# y = [11.0, 11.1, 12.5, 8.4, 9.3, 8.7, 6.4, 8.5, 7.8, 9.1,
#      8.2, 12.2, 11.9, 9.6, 10.9, 9.6, 10.1, 8.1, 6.8, 8.9,
#      7.7, 8.5, 8.9, 10.4, 11.1]
# model = SLR(len(x), x, y)
# print(model.parameters)
# print(model.CI_parameters())
# print(model.significance_test())

# 25.4
# N = [3, 3, 3, 3, 3]
# X = [30, 40, 50, 60, 70]
# Y = [[13.7, 14.0, 14.6], [15.5, 16.0, 17.0], [18.5, 20.0, 21.1], [17.7, 18.1, 18.5], [15.0, 15.6, 16.5]]
# print(LoF_test(N, X, Y))

# N = [2, 2, 3, 3, 2, 2]
# X = [1.0, 3.3, 4.0, 5.6, 6.0, 6.5]
# Y = [[1.6, 1.8], [1.8, 2.7], [2.6, 2.6, 2.2], [3.5, 2.8, 2.1], [3.4, 3.2], [3.4, 3.9]]
# print(LoF_test(N, X, Y))

X = [100, 120, 140, 160, 180]
Y = [45, 54, 66, 74, 85]
model = SLR(5, X, Y)
print(model.CI(130))
print(model.PI(130))


'''
to do: 
slides 595, plot for CI and PI
slides 601, Test for Correlation
slides 614, residual plot
'''
