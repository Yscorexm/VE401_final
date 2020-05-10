import numpy as np
from scipy import stats
import matplotlib.pyplot as plot

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
        self.sum_x=sum(x)
        self.sum_y=sum(y)
        self.sum_x2 = sum([k**2 for k in x])
        self.sum_y2 = sum([k**2 for k in y])


        self.sum_xy = sum(np.multiply(x,y))

        
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
        # self.e = [y[i] - self.predict(x[i]) for i in range(n)]


    def regression(self):
        print("The related parameters are:")
        print("n=", self.n)
        print("sum_x=", self.sum_x)
        print("sum_y=", self.sum_y)
        print("sum_x2=", self.sum_x2)
        print("sum_y2=", self.sum_y2)
        print("sum_xy=", self.sum_xy)
        print("===============================")
        print("Sxx=",self.Sxx)
        print("Syy=",self.Syy)
        print("Sxy=",self.Sxy)
        print("===============================")
        print("(b0,b1)=",self.parameters)

    def predict(self, x_0):
        '''
        RETURN: Estimated Mean at x = x_0.
        '''
        prediction=self.b0+self.b1*x_0
        return prediction

    def unmute_predict(self,x_0):
        print("parameters (b0,b1)=", self.parameters)
        prediction = self.predict(x_0)
        print("prediction at x0=", x_0, "= b0+b1*x0=", prediction)
        return prediction


    def CI_parameters(self, alpha=0.05):
        '''
        Confidence Intervals on the Slope and the Intercept.  
        In slides 580.  
        RETURN: CI tuple, first for β0 and second for β1.
        '''
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        print("t(alpha/2,n-2)=",t)
        diff_1 = t * self.S / self.Sxx**0.5

        diff_0 = t * self.S * (self.sum_x2 / (self.n * self.Sxx))**0.5
        print("diff(beta_0)=",diff_0)

        print("diff(beta_1)=", diff_1)
        CI_beta0 = (self.b0 - diff_0, self.b0 + diff_0)
        CI_beta1 = (self.b1 - diff_1, self.b1 + diff_1)
        print("The CI for beta0 and beta1:")
        print("beta0=b0±t(alpha/2, n-2)*S*sqrt(sum_x2)/sqrt(n*Sxx)=", CI_beta0)
        print("beta1=b1±t(alpha/2, n-2)*S*sqrt(sum_x2)/sqrt(n*Sxx)=", CI_beta1)
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
        print("This is the significance test. H0: beta_1=0")
        print("T=",T,"      t=",t)
        if abs(T)>t:
            print("Reject!=> β1 ≠ 0")
        else:
            print("Cannot reject!")
        return T, t

    def CI(self, x_0, alpha=0.05):
        '''
        Confidence Interval for the Estimated Mean at x = x_0.  
        In slides 589.  
        REQUIRE: x_0.  
        RETURN: CI.  
        IMPORTANT: to be tested.
        '''
        print("This is confidence interval with alpha=",alpha)
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        mean = self.predict(x_0)
        print("mean=",mean)
        diff = t * self.S * (1/self.n + (x_0 - self.x_mean)**2/self.Sxx)**0.5
        print("diff=",diff)
        CI=(mean-diff, mean+diff)
        print("CI=",CI)
        return mean-diff, mean+diff
    
    def PI(self, x_0, alpha=0.05):
        '''
        Prediction Interval for the observed value at x = x_0.  
        In slides 593.  
        REQUIRE: x_0.  
        RETURN: PI.
        IMPORTANT: to be tested.
        '''

        print("This is prediction interval with alpha=",alpha)
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        mean = self.predict(x_0)
        print("mean=", mean)
        diff = t * self.S * (1 + 1/self.n + (x_0 - self.x_mean)**2/self.Sxx)**0.5
        print("diff=", diff)
        PI=(mean-diff, mean+diff)
        print("PI=", PI)
        return mean-diff, mean+diff
    
    def correlation_test(self, alpha=0.05):
        '''
        Test for Correlation.  
        H0: ϱ = 0.  
        In slides 601.  
        RETURN: Test statistics and critical value.  
        IMPORTANT: to be tested.
        '''
        print("This is the test for correlation. H0: ϱ = 0")
        print("R2=",self.R_square,"      R=",self.R_square**0.5)
        t = stats.t(self.n - 2).ppf(1-alpha/2)
        T = ((self.R_square) * (self.n - 2) / (1 - self.R_square))**0.5

        print("T=",T, "         t=",t)

        if abs(T)>t:
            print("Reject!=> ϱ ≠ 0")
        else:
            print("Cannot reject!")

        return T, t

    def residue_plot(self):
        e = [self.y[i] - self.predict(self.x[i]) for i in range(self.n)]
        print("This is the residual plot. The residuals are:")
        print(e)

        fig=plot.figure(num=1, figsize=(15, 8),dpi=80)
        plot.scatter(self.x,e)
        plot.show()

    

def LoF_test(N, X, Y, alpha=0.05):
    '''
    Test for Lack of Fit.  
    H0: the linear regression model is appropriate.  
    In slides 608.  
    REQUIRE: multiple sampling for single x. N, X are 1-D lists. Y is a 2-D list.  
    RETURN: Test statistics and critical value.
    '''
    k = len(N)
    n = sum(N)
    print("This is lack of fit test. H0: model is appropriate.")
    print("k=",k)
    print("n=",n)

    mean_Y = [sum(Y[i])/N[i] for i in range(k)]
    SSE_pe = sum(sum([(Y[i][j] - mean_Y[i])**2 for j in range(N[i])]) for i in range(k))

    print("SSE_pe=",SSE_pe)
    x = []
    for i in range(k):
        x.extend([X[i]] * N[i])
    y = []
    for i in range(k):
        y.extend(Y[i])
    model = SLR(n, x, y)
    SSE = model.SSE
    SSE_lf = SSE - SSE_pe
    print("SSE=", SSE)
    print("SSE_lf=SSE-SSE_pe=", SSE_lf)
    F = (SSE_lf / (k - 2)) / (SSE_pe / (n - k))
    f = stats.f(k - 2, n - k).ppf(1-alpha)
    print("F=(SSE_lf / (k - 2)) / (SSE_pe / (n - k))=", F)

    print("f(k-2,n-k,alpha)=", f)

    if F > f:
        print("Reject!=> model is *not* appropriate")
    else:
        print("Cannot reject!=> maybe appropriate")
    return F, f






# 24.1 24.5 24.7
# x = [35.3, 27.7, 30.8, 58.8, 61.4, 71.3, 74.4, 76.7, 70.7, 57.5,
#      46.4, 28.9, 28.1, 39.1, 46.8, 48.5, 59.3, 70.0, 70.0, 74.4,
#      72.1, 58.1, 44.6, 33.4, 28.6]
# y = [11.0, 11.1, 12.5, 8.4, 9.3, 8.7, 6.4, 8.5, 7.8, 9.1,
#      8.2, 12.2, 11.9, 9.6, 10.9, 9.6, 10.1, 8.1, 6.8, 8.9,
#      7.7, 8.5, 8.9, 10.4, 11.1]
# model = SLR(len(x), x, y)
# model.regression()  #24.1
# model.CI_parameters()  #24.5
# # # model.predict(5)
# model.significance_test() #24.7
# model.residue_plot()
# model.significance_test()
# model.correlation_test()
# print(model.CI_parameters())
# print(model.significance_test())

# 25.4
# N = [2,2,3,3,2,2]
# X = [1.0,3.3,4.0,5.6,6.0,6.5]
# Y = [[1.6,1.8], [1.8,2.7], [2.6,2.6,2.2], [3.5,2.8,2.1], [3.4,3.2],[3.4,3.9]]
# print(LoF_test(N, X, Y))

# X = [100,120,140,160,180]
# Y= [45,54,66,74,85]
#
#
# model=SLR(len(X),X,Y)
# model.regression()
# model.CI_parameters()
# model.CI(130)
# model.PI(130)
# model.residue_plot()
'''
to do: 
slides 595, plot for CI and PI 不用做吧
slides 601, Test for Correlation 做好了
slides 614, residual plot

如果有1个x对多个y的情况只需输入各对(x,y)即可
'''
