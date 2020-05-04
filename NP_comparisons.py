from scipy import stats

def wrs_test(x, y, H0, alpha=0.05):
    '''
    Wilcoxon Rank-Sum Test.  
    H0: see REQUIRE.  
    In slides 490-491.  
    REQUIRE: len(x) <= len(y).  
    H0 can take two values: "less" or "greater".  
    "less" means P[x > y] <= 1/2, while "greater" means P[x > y] >= 1/2.  
    RETURN: (w_x, E[w_x], Var[w_x]), (Test statistics, p-value).  
    '''
    # def get_median(d):
    #     n = len(d)
    #     if n % 2 == 0:
    #         median = (d[n//2 - 1] + d[n//2]) / 2
    #     else:
    #         median = d[n//2]
    #     return median
    
    # process_x = [(k, 1) for k in x]
    # process_y = [(k, 2) for k in y]
    # data = process_x + process_y
    # if median == None:
    #     print("Median:", get_median([k[0] for k in data]))
    #     return

    process_x = [(k, 1) for k in x]
    process_y = [(k, 2) for k in y]
    m, n = len(process_x), len(process_y)
    data = process_x + process_y
    n = len(data)
    data.sort(key=lambda x: x[0])

    rank_list = []
    last = -2134444
    cnt = 0
    sum_of_ties = 0
    for i in range(n):
        if data[i][0] == last:
            cnt += 1
        else:
            rank = (2*i - cnt + 1)/2
            for _ in range(cnt):
                rank_list.append(rank)
            if cnt != 1:
                sum_of_ties += (cnt**3 + cnt) / 12
            last = data[i][0]
            cnt = 1
    else:
        rank = (2*i - cnt + 3)/2
        for _ in range(cnt):
            rank_list.append(rank)
    # print([(data[i][1], data[i][0], rank_list[i]) for i in range(n)])
    w_x = sum([rank_list[i] for i in range(n) if data[i][1] == 1])
    n -= m
    E_w_x = m*(m+n+1)/2
    if sum_of_ties > 12:
        print("Varning: sum of ties = ", sum_of_ties, ", which is > 12.")
        Var_w_x = m*n*(m+n+1)/12
    else:
        print("sum of ties = ", sum_of_ties)
        Var_w_x = m*n*(m+n+1)/(12-sum_of_ties)
    Z = (w_x - E_w_x)/Var_w_x**0.5
    if H0 == "less":
        p_value = 1 - stats.norm.cdf(Z)
    elif H0 == "greater":
        p_value = stats.norm.cdf(Z)
    return (w_x, E_w_x, Var_w_x), (Z, p_value)


# 22.2
y = [5.5, 5.5, 12.75, 18.75, 19.25, 11.25,
     11.5, 11.5, 12.25, 14.25, 9.25, 14.5,
     13.25, 8.25, 16.75, 10.5, 6, 15.25,
     6.5, 12.5, 10.5, 8.75, 11.5, 17,
     2.75, 13.25, 19, 16.5, 11.5, 1.75]
x = [18.5, 12.25, 3, 15, 19.75, 11.25, 
     11.75, 19.25, 12.25, 19.75, 16.25, 13,
     19.25, 1.75]
print(wrs_test(x, y, "less"))