import numpy as np 
from scipy import stats


class MLR:
    '''
    multiple linear regression  
    Y = Xb + E
    '''

    def __init__(self, n, p, X, Y):
        self.n = n
        self.p = p
        self.X = np.array([[1]*n] + X).T
        self.Y = np.array(Y)
        self.b = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        self.E = self.Y - self.X @ self.b
        self.SSE = self.E.T @ self.E
        self.P = np.ones((n, n)) / n
        self.SST = self.Y.T @ ((np.eye(n) - self.P)@Y)
        self.H = self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T
        self.SSR = self.SST - self.SSE
        self.Rsquare = self.SSR / self.SST
        self.S = (self.SSE / (n-p-1))**0.5
    
    def significance_test(self, alpha=0.05):
        '''
        F-Test for Significance of Regression.  
        H0: β1 = β2 = ··· = βp = 0.  
        In slides 656.  
        RETURN: Test statistics and critical value.  
        '''
        # test_F = self.SSR/self.p/self.S**2
        # or
        test_F = (self.n-self.p-1)/self.p * self.Rsquare/(1-self.Rsquare)
        f = stats.f(self.p, self.n-self.p-1).ppf(1-alpha)
        return test_F, f
    
    def CI_parameters(self, alpha=0.05):
        '''
        Confidence Intervals for the Model Parameters.  
        In slides 664.  
        RETURN: CI matrix, one row for one parameter.  
        '''
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        sqrt_xi = np.array([XTX_inv[j, j]**0.5 for j in range(0, self.p+1)])
        t = stats.t(self.n-self.p-1).ppf(1-alpha/2)
        lower_bound = self.b - t*self.S*sqrt_xi
        upper_bound = self.b + t*self.S*sqrt_xi
        CI = np.array([lower_bound, upper_bound]).T
        return CI
    
    def CI(self, x, alpha=0.05):
        '''
        Confidence Intervals for the Estimated Mean.  
        In slides 670.  
        REQUIRE: x is a list without the first element 1.  
        RETURN: CI.
        '''
        t = stats.t(self.n-self.p-1).ppf(1-alpha/2)
        x_0 = np.array([1] + x)
        u_Y_x_0 = x_0.T @ self.b
        diff = t * self.S * (x_0.T @ np.linalg.inv(self.X.T @ self.X) @ x_0)**0.5
        return u_Y_x_0 - diff, u_Y_x_0 + diff
    
    def TMS_test(self, j, alpha=0.05):
        '''
        T-Test for Model Sufficiency.  
        H0: βj = 0.  
        In slides 672.  
        REQUIRE: j = 0, 1, 2,..., p.
        RETURN: Test statistics and critical value.
        '''
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        sqrt_xi = np.array([XTX_inv[j, j]**0.5 for j in range(0, self.p+1)])
        T = self.b[j] / self.S / sqrt_xi[j]
        t = stats.t(self.n-self.p-1).ppf(1-alpha/2)
        return T, t
    




def FMS_test(full, reduced, alpha=0.05):
    '''
    Partial F-Test for Model Sufficiency.  
    H0: the reduced model is sufficient.  
    In slides 678.  
    REQUIRE: full and reduced is of MLR class.  
    RETURN: Test statistics and critical value.  
    '''
    F = (full.n - full.p - 1)/ (full.p - reduced.p) * (reduced.SSE - full.SSE)/full.SSE
    f = stats.f(full.p - reduced.p, full.n - full.p - 1).ppf(1-alpha)
    return F, f





# 26.1
# x = [5, 5, 10, 10, 15, 15, 20, 20, 25, 25]
# x_square = [k**2 for k in x]
# y = [14.0, 12.5, 7.0, 5.0, 2.1, 1.8, 6.2, 4.9, 13.2, 14.6]
# model = MLR(10, 2, [x, x_square], y)

# 26.2 27.6 27.11
# x1 = [1.35, 1.90, 1.70, 1.80, 1.30, 2.05, 1.60, 1.80, 1.85, 1.40]
# x2 = [90, 30, 80, 40, 35, 45, 50, 60, 65, 30]
# y = [17.9, 16.5, 16.4, 16.8, 18.8, 15.5, 17.5, 16.4, 15.9, 18.3]
# model = MLR(len(x1), 2, [x1, x2], y)
# print(model.CI([1.5, 70]))

# 27.13 27.15
# x = [5, 7.5, 10, 12.5, 15, 17.5, 20]
# x_square = [k**2 for k in x]
# y = [1, 2.2, 4.9, 5.3, 8.2, 10.7, 13.2]
# model = MLR(len(x), 2, [x, x_square], y)
# reduced_model = MLR(len(x), 1, [x], y)
# print(model.TMS_test(0))



x=[0.50, 1.00, 1.50, 2.00, 2.50]
y=[-0.51 ,-2.09 ,-6.03 ,-9.28 ,-17.12]
x_square = [k**2 for k in x]
model = MLR(len(x), 2, [x, x_square], y)
print(model.significance_test())
reduced_model = MLR(len(x), 1, [x], y)
print(FMS_test(model, reduced_model))
