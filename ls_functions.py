import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import warnings
import scipy.stats as stats
from scipy.stats import levene

#function definition
def membership_function(data, var_name, value, central=0, spread=0.1, plot=False, na_omit=False, 
           expert = False,use_central_and_spread=False):
    d = deepcopy(data)
    
    if na_omit:
        d = d.loc[~d[var_name].isna()]
    else:
        d = d.fillna(0)
        
    d = d[var_name]
    
    max_for_universe = np.max(d)  
    min_for_universe = np.min(d)
    
    universe = np.arange(min_for_universe, max_for_universe, 0.001)
    
    reg_name = var_name 
    
    reg = ctrl.Consequent(universe, reg_name)

    if use_central_and_spread:
        first_quartile = np.max([central-(spread),min_for_universe])
        median_quartile = central
        third_quartile = np.min([central+(spread),max_for_universe])
    else:        
        first_quartile = np.percentile(d, 25)
        median_quartile = np.percentile(d, 50)
        third_quartile = np.percentile(d, 75)
        
   #quartiles based fuzzification
    low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
    medium = fuzz.trimf(reg.universe, [first_quartile, median_quartile, third_quartile])
    high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
     
    if plot:     
        fig, (ax0) = plt.subplots(nrows=1, figsize=(5, 3))
        ax0.plot(universe, low, 'b', linewidth=2, label='low')
        ax0.plot(universe, medium, 'r', linewidth=2, label='medium')
        ax0.plot(universe, high, 'g', linewidth=2, label='high')
        ax0.set_title(str(var_name))
        ax0.legend()
        plt.tight_layout()
        plt.close()
        fig.savefig("LinguisticVariable_"+str(var_name)+"_spread_"+str(spread)+".png")
        #quit()

    return (fuzz.interp_membership(universe, low, value),
            fuzz.interp_membership(universe, medium, value),
            fuzz.interp_membership(universe, high, value)
            )

#Test degrees    
def calculate_membership(data, var_name, plot=False, na_omit=True, expert=False, printout=False):
    column = data[var_name]
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
    
    #for i in range(1):
    for i in range(len(column)):
        result.loc[i,] = membership_function(data, var_name, column[i], 0, 0, plot, na_omit, expert)
        if printout==True:
            print(str(result.loc[i,]))
            print(str(column[i]))

            
    return result

def calculate_membership_fixed(data, var_name, plot=False, na_omit=True, expert=False, printout=False,
                              use_central_and_spread=True, central=0, spread=0.1):
    column = data[var_name]
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
    
    #for i in range(1):
    for i in range(len(column)):
        result.loc[i,] = membership_function(data, var_name, column[i], central, spread, 
                  plot, na_omit, expert, use_central_and_spread=True) 
        if printout==True:
            print(str(result.loc[i,]))
            print(str(column[i]))

            
    return result

def quantifier(x):
    part = np.arange(0, 1.01, 0.001)
    majority = fuzz.trapmf(part, [0.5, 0.6, 1, 1])
    minority = fuzz.trapmf(part, [0, 0, 0.3, 0.50])
    almost_all = fuzz.trapmf(part, [0.8, 0.9, 1, 1])
    part_majority = fuzz.interp_membership(part, majority, x)
    part_minority = fuzz.interp_membership(part, minority, x)
    part_almost_all =  fuzz.interp_membership(part, almost_all, x)
    return dict(majority = part_majority, 
                minority = part_minority, 
                almost_all = part_almost_all)
   

def t_norm(a, b, ntype):
    """
    calculates t-norm for param a and b
    :param ntype:
        1 - minimum
        2 - product
        3 - Lukasiewicz t-norm
    """
    if ntype == 1:
        return np.minimum(a, b)
    elif ntype == 2:
        return a * b
    elif ntype == 3:
        return np.maximum(0, a + b - 1)

def Degree_of_truth(d, Q="majority", P="loudness_low", P2=""):
    """
    Degree of truth for short protoforms
    """
    p = np.mean(d[P])

    return quantifier(p)[Q]

def Degree_of_truth_ext(d, Q="majority", P="loudness_medium", R="", tnorm="min"):
    """
    Degree of truth for extended protoforms
    """

    if(tnorm=="min"):
        p = np.fmin(d[P], d[R])
    else:
        p = np.fmax(0,(d[P]+d[R]-1))
    
    r = d[R]
    t = np.sum(p) / np.sum(r)
    
    if np.sum(r) == 0:
        t = 0
    else:
        t = np.sum(p) / np.sum(r)

    return quantifier(t)[Q]

def Degree_of_support(d, P="loudness_medium"):
    """
    Degree of support for short protoforms informs how many objects are covered by a particular summary
    """

    DoS = sum(d[P]>0)/len(d)
    
    return DoS

def Degree_of_support_ext(d, P="loudness_medium", R="quality_low", tnorm="min"):
    """
    Degree of support for extended protoforms informs how many objects are covered by a particular summary
    """

    if(tnorm=="min"):
        p = np.fmin(d[P], d[R])
    else:
        p = np.fmax(0,(d[P]+d[R]-1))
    
    DoS = sum(p>0)/len(d)
    
    return DoS

def Degree_of_focus_ext(d, P="loudness_medium", R="quality_low"):
    """
    Degree of focus applies to extended protoforms and informs how many objects satisfy the qualifier of the particular summary
    """
    DoF = sum(d[R])/len(d)
    
    return DoF

def all_protoform(d, var_names, Q = "majority", desc = 'most', classtoprint="class"):
    """
    Function that determines the degrees of truth support and focus for all linguistic summaries (simple and complex)   
    """
    
    pp = [var_names[0] + "_low", var_names[0] + "_medium", var_names[0] + "_high"]
    qq = [var_names[1] + "_low", var_names[1] + "_medium", var_names[1] + "_high"]
    qq_shap_print = ["against predicting "+classtoprint+" class", "around zero to predicting "+classtoprint +" class", "positively to predicting "+classtoprint+" class"]
    pp_print = [var_names[0], var_names[0],var_names[0]]
    pp_print1 = ["low", "medium","high"]
    
    protoform = np.empty(9, dtype = "object")
    Id = np.zeros(9)
    DoT = np.zeros(9)
    DoS = np.zeros(9)
    DoF = np.zeros(9)
    k = 0
    
    for i in range(len(pp)):
        for j in range(len(qq)):   
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, P = pp[j], R = qq[i])
            DoF[k] = Degree_of_focus_ext(d = d, P = pp[j], R = qq[i])
            protoform[k] = "Among records that contribute "+ qq_shap_print[i] + ", "+ desc + " of them have " + pp_print[j] + "-related features at "+pp_print1[j]+" level."
            Id[k] = k
            k += 1
            
    dd = {'Id': Id,
          'protoform': protoform,
            'DoT': DoT,
            'DoS': DoS,
            'DoF' : DoF}
    dd = pd.DataFrame(dd)   
    return dd[['Id', 'protoform', "DoT", 'DoS', "DoF"]]


def ls_ind_params(data,shapdata, plot=False, expert=False, printout=False, spread=0.1, classtoprint='0'):
    acoustic_var_names = data.columns
    df_protoform_all = []
    for name in acoustic_var_names:
        temp = calculate_membership(data, name, plot,expert=expert, printout=printout)
        temp2 = calculate_membership_fixed(shapdata, name, plot,expert=expert, 
                                              printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=['shap_'+name+'_low','shap_'+name+'_medium','shap_'+name+'_high']
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        var_names=[name, "shap_"+name]
        df_protoform = all_protoform(data_for_lingsum, var_names, Q = 'majority', desc = 'most', classtoprint=classtoprint)
        df_protoform_all.append(df_protoform)
    df_protoform_all = pd.concat(df_protoform_all)
    return(df_protoform_all)


def ls_group_params(acoustic_group,acoustic_group_label, data,shapdata, plot=False, expert=False, printout=False, spread=0.1, classtoprint='0',model="xxx"):
    data_for_lingsum_all = []
    if (classtoprint == 2): #dla manii
        spread=0.02
    else: spread=0.1
    for name in acoustic_group:  
        temp = calculate_membership(data, name, plot,expert=expert, printout=printout)
        temp.columns=[acoustic_group_label+'_low',acoustic_group_label+'_medium',acoustic_group_label+'_high']
        temp2 = calculate_membership_fixed(shapdata, name, plot,expert=expert, printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=['shap_'+acoustic_group_label+'_low','shap_'+acoustic_group_label+'_medium','shap_'+acoustic_group_label+'_high']
        filenametemp = model+"temp_"+acoustic_group_label+"_"+ classtoprint + ".csv"
        filenametemp2 = model+"temp2_"+acoustic_group_label+"_"+classtoprint + ".csv"
        temp.to_csv(filenametemp)
        temp2.to_csv(filenametemp2)
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        var_names=[acoustic_group_label, 'shap_'+acoustic_group_label]
        data_for_lingsum_all.append(data_for_lingsum)

    data_for_lingsum_all = pd.concat(data_for_lingsum_all )
    df_protoform = all_protoform(data_for_lingsum_all, var_names, Q = 'majority', desc = 'most', classtoprint=classtoprint)
    return df_protoform


def round_with_padding(value, round_digits):
    return format(round(value, round_digits), "."+str(round_digits)+"f")

def is_statistically_different_via_test(np_array_data_group1, np_array_data_group2):
    """
    Mann Whitney U-Test
    
    It requires several assumptions:
    1. Assumption #1: You have one dependent variable that is measured at the continuous or ordinal level. 
    2. You have one independent variable that consists of two categorical, independent groups
    3. You should have independence of observations, which means that there is no relationship between the observations in each group of the independent variable or between the groups themselves.
    4. Homogeneity assumption: Mann-Whitney U test can be used when your two variables are not normally distributed. However, 
    to compare medians the distribution of both samples must have the same shape (including dispersion)

    Example of use:
    a=np.array([[1, 2, 3], [4, 5, 6]])
    b=np.array([1, 2, 5])
    is_statistically_different(data['a'], data['b'])
    """

    # Print the variance of both data groups
    print("Applying the Mann-Whitney U test to determine whether one group has higher or lower scores than the other group.")
    print("\n Variances of both groups: ", round_with_padding(np.var(np_array_data_group1), 2), round_with_padding(np.var(np_array_data_group2), 2))
    stat, p_value_variances = levene(np_array_data_group1, np_array_data_group2)
    if p_value_variances < 0.05:
        print("WARNING: The p-value ", p_value_variances, " is too small (<0.05) and suggests that the test cannot be applied, because the populations do not have equal variances" )
        print("[Any interpretation of differences between groups becomes difficult when variances are not equal!]" )
        # Potential alternatives for the future:
        # https://www.researchgate.net/publication/240444870_Mann-Whitney_U_test_when_variances_are_unequal
    else:
        print("\n Levene test for equal variance populations passed! p-value: ", p_value_variances)

    res, p_value = stats.mannwhitneyu(np_array_data_group1, np_array_data_group2)
    print("\n U statistic", stat, " p-value: ", p_value)


    # Statistical difference significance levels
    if p_value < 0.001:
        print("Yes, for a statistically significance significance level alpha = 0.001  ")
    elif p_value < 0.01:
        print("Yes, for a statistically significance significance level alpha = 0.01  ")
    elif p_value < 0.05:
        print("Yes, for a statistically significance significance level alpha = 0.05  ")
    else:
        print("No, the samples are not statistically different")