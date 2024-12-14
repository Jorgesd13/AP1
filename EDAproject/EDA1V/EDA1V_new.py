#!/usr/bin/env python
# coding: utf-8
def func_preprocessing():
    # # A. IMPORT LIBRARIES
    
    # In[3]:
    
    
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from scipy import stats as st
    from scipy.stats import norm, beta
    from sklearn.mixture import GaussianMixture
    import pickle
    
    
    # # B. IMPORT DATASETS
    
    # In[4]:
    
    
    ####CHECK STORED DF#####
    import os
    # Save the current working directory
    saved_path = os.getcwd()
    #print(f"Current path saved: {saved_path}")
    
    # Change the current working directory to a specified path (e.g., "/path/to/directory")
    new_path = r"C:\Users\carlo\Desktop\Projects\AP1\AP1_repo\EDAproject\EDA1V"
    os.chdir(new_path)
    #print(f"Changed to new path: {os.getcwd()}")

    # Deserialization
    with open("dfXY.pickle", "rb") as infile:
        dfXYPickle = pickle.load(infile)
    #print("Reconstructed object", dfXYPickle)
    
    # Change the current working directory back to the saved path
    os.chdir(saved_path)
    #print(f"Back to saved path: {os.getcwd()}")
    
    # In[5]:
    
    
    dfXY = dfXYPickle.copy()
    
    
    #  # C. Pre-process
    # 
    # 
    
    # In[14]:
    #Strip column names
    dfXY.columns=dfXY.columns.str.strip(' \t\n')
    dfXY.drop(["SteelFamCluster"],axis=1,inplace=True)
    
    # ## C.1 Clean categorical vars
    
    # In[19]:
    
    
    #Remove decimals in categorical
    dfXY[dfXY.columns[dfXY.dtypes == "category"]]=dfXY[dfXY.columns[dfXY.dtypes == "category"]].astype(np.int64)
    dfXY[["AP1_FurnaceWeldInFurnace","AP1_FurnaceZone1TurndownOn","AP1_FurnaceZone2TurndownOn","AP1_FurnaceZone3TurndownOn","AP1_FurnaceZone4TurndownOn","AP1_FurnaceZone5TurndownOn","AP1_FurnaceZone6TurndownOn","AP1_FurnaceZone7TurndownOn","AP1_FurnaceZone8TurndownOn","grade"]]=dfXY[["AP1_FurnaceWeldInFurnace","AP1_FurnaceZone1TurndownOn","AP1_FurnaceZone2TurndownOn","AP1_FurnaceZone3TurndownOn","AP1_FurnaceZone4TurndownOn","AP1_FurnaceZone5TurndownOn","AP1_FurnaceZone6TurndownOn","AP1_FurnaceZone7TurndownOn","AP1_FurnaceZone8TurndownOn","grade"]].astype("category")#,"SteelGradeID"

    # In[20]:
    #Make a copy of dfXY in case
    dfXYold = dfXY.copy()
    
    # ## C.2 Clean numeric vars
    
    # In[21]:
    
    
    # Numeric variable .description() check
    # 
    # min/max vs quantiles, mean, and standard deviation – looking for outliers/sensible results
    # 
    # Distribution plot examination
    # 
    # can use .hist() command or a more customized subplot group or for-each loop to plot a seperate histogram of each variable
    
    
    # In[22]:
   #Numeric vars
    numerical_var_series=dfXY[dfXY.columns[dfXY.dtypes == np.float64]].columns#.to_list()#[0:5]
    
    # In[23]:
    #Create the different groups of columns to plot
    proceso_vars_series=dfXY[numerical_var_series[(~numerical_var_series.str.contains("AP1_Furnace",regex=False)) & (~numerical_var_series.str.contains("pyro2",regex=False))]].columns#.head(1)
    zoneTemp_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Temp.*",regex=True)]].columns#.head(1)
    
    zoneAirGas_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Air.*|.*Zone.*Gas.*",regex=True)]].columns#.head(1)
    
    recuperator_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains("AP1_Furnace.*Dilution|Recuperator|RWP|Combustion.*",regex=True)]].columns#.head(1)
    
    pyro_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*pyro.*|.*Pyro.*",regex=True)]].columns#.head(1)
    
    rest_vars_series=dfXY[[var for var in numerical_var_series.to_list() if var not in proceso_vars_series.to_list() and var not in zoneTemp_vars_series.to_list() and var not in zoneAirGas_vars_series.to_list() and var not in recuperator_vars_series.to_list() and var not in pyro_vars_series.to_list()]].columns#.head(1)
    
    
    # In[24]:
    #Statistic description of numeric vars
    dfXYold_description=dfXYold.describe()
    
    
    # In[25]:
    #Cuts to consider outliers
    cut_neg = dfXYold_description.iloc[1] - 3*dfXYold_description.iloc[2] #mean-3*std
    cut_pos = dfXYold_description.iloc[1] + 3*dfXYold_description.iloc[2] #mean+3*std
    cut_neg.name = 'cut-'
    cut_pos.name = 'cut+'
    cut_neg = pd.DataFrame(cut_neg).T
    cut_pos = pd.DataFrame(cut_pos).T
    tbl_desc = pd.concat([dfXYold_description,cut_neg, cut_pos])
    tbl_desc
    
    
    # dfXY=dfXYold.copy()
    
    # In[26]:
    #Ini Tune cut
    for c in range(tbl_desc.shape[1]):
        if tbl_desc.loc['cut-',tbl_desc.columns[c]]<0:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
        if tbl_desc.columns[c]=='AP1_FurnaceWidth':
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0 #some widthIDs are smaller than the actual cut-
        elif tbl_desc.columns[c]=='NetWeight':
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=5000 #min around 20k, if cut for defects would be 10k. To be conservative 5k
        elif tbl_desc.columns[c] in tbl_desc.columns[tbl_desc.columns.str.contains('Ratio',regex=False)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
            tbl_desc.loc['cut+',tbl_desc.columns[c]]=4
        elif tbl_desc.columns[c] in tbl_desc.columns[tbl_desc.columns.str.contains("P1_FurnaceZone.*[C].*[V].*",regex=True)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
            tbl_desc.loc['cut+',tbl_desc.columns[c]]=100 #CV vars are in %
        elif  tbl_desc.columns[c] in zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains("Zone0",regex=False) & ~zoneAirGas_vars_series.str.contains("Ratio",regex=False)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
            tbl_desc.loc['cut+',tbl_desc.columns[c]]=100 #Zone0 airgas vars are in %
        elif tbl_desc.columns[c] in zoneTemp_vars_series[~zoneTemp_vars_series.str.contains("CV",regex=False)]:
            if tbl_desc.loc['cut-',tbl_desc.columns[c]]<=500:
                tbl_desc.loc['cut-',tbl_desc.columns[c]]=500 #Zone 0 seems to have many values near 0 that are outside the bell
        elif tbl_desc.columns[c] in pyro_vars_series[pyro_vars_series.str.contains("Temp",regex=False)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=1400 #pyro sensors seem to have many values at 1200 that are outside the bimodal bell, p() bc it is the lower limit of the sensor    
    
    # In[27]:

    # #### Make Air/Gas Flows in 0-100%
    # Air/Gas ratios are in 0-4 scale
    # Zone0: Already in %
    # Zone1: Bottom/Top Air Flow in 0-47565 ($59655$)SCFH scale, Bottom/Top Gas Flow in 0-5170 SCFH scale
    # Zone2: Bottom/Top Air Flow in 0-47565 ($59655$)SCFH scale, Bottom/Top Gas Flow in 0-5170 SCFH scale
    # Zone3: Bottom/Top Air Flow in 0-34880 ($52831$)SCFH scale, Bottom/Top Gas Flow in 0-3350 SCFH scale
    # Zone4: Bottom/Top Air Flow in 0-34880 ($51951$)SCFH scale, Bottom/Top Gas Flow in 0-3685 SCFH scale
    # Zone5: Bottom/Top Air Flow in 0-27070 ($40318$)SCFH scale, Bottom/Top Gas Flow in 0-2860 SCFH scale
    # Zone6: Bottom/Top Air Flow in 0-27070 ($41004$)SCFH scale, Bottom/Top Gas Flow in 0-2860 SCFH scale
    # Zone7: Bottom Air Flow in 0-17490 ($26050$)SCFH scale, Bottom Gas Flow in 0-1680 SCFH scale
    #        Top Air Flow in 0-20400 ($30384$)SCFH scale, Top Gas Flow in 0-1960 SCFH scale
    # Zone8: Bottom Air Flow in 0-20400 ($30384$)SCFH scale, Bottom Gas Flow in 0-1960 SCFH scale
    #        Top Air Flow in 0-17490 ($26050$)SCFH scale, Top Gas Flow in 0-1680 SCFH scale
    # 
    
    # #AirGasRatios are already adimensional so no need of scaling
    # dfXY.loc[:,dfXY.columns[dfXY.columns.str.contains("Ratio",regex=False)]] = dfXY.loc[:,dfXY.columns[dfXY.columns.str.contains("Ratio",regex=False)]]/4*100
    
    # In[28]:
    
    
    #Zone1
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/59655*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/5170*100
    
    
    # In[29]:
    
    
    #Zone2
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/59655*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/5170*100
    
    
    # In[30]:
    
    
    #Zone3
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/52831*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/3350*100
    
    
    # In[31]:
    
    
    #Zone4
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/51951*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/3685*100
    
    
    # In[32]:
    
    
    #Zone5
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/40318*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/2860*100
    
    
    # In[33]:
    
    
    #Zone6
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/41004*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/2860*100
    
    
    # In[34]:
    
    
    #Zone7
    #AirBottom
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7BottomAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7BottomAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7BottomAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/26050*100
    #AirTop
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/30384*100
    #GasBottom
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7BottomGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7BottomGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7BottomGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/1680*100
    #GasTop
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone7TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/1960*100
    
    
    # In[35]:
    
    
    #Zone8
    #AirBottom
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8BottomAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8BottomAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8BottomAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/30384*100
    #AirTop
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/26050*100
    #GasBottom
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8BottomGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8BottomGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8BottomGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/1960*100
    #GasTop
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone8TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/1680*100
    
    
    
    # ## C.3 Create sort of SteelFamID: clustersGMM
    
    # #Create sort of SteelFamID
    # #CLUSTERING PYRO2GOALS
    # from sklearn.cluster import DBSCAN
    # from sklearn.mixture import GaussianMixture
    # clustersDBSCAN = DBSCAN(eps=50, min_samples=5).fit_predict(dfXY.pyro2.to_numpy().reshape(-1,1))
    # print(np.unique(clustersDBSCAN))
    # clustersGMM = GaussianMixture(n_components=2).fit(dfXY.pyro2.to_numpy().reshape(-1,1)).predict(dfXY.pyro2.to_numpy().reshape(-1,1))
    # print(np.unique(clustersGMM))
    # 
    # fig,axes = plt.subplots(1,2,figsize=(15,6))
    # fig.suptitle("Clustering",fontsize=15)
    # clusteringDBSCAN=pd.merge(right=pd.concat([pd.DataFrame(dfXY.pyro2).reset_index(drop=True),pd.DataFrame(clustersDBSCAN,columns=['clustersDBSCAN'])],axis=1).drop_duplicates(),left=pd.DataFrame(dfXY.pyro2.value_counts()).reset_index(), how='inner',on='pyro2')
    # #sns.scatterplot(data=clustering,x='pyro2',y='count',hue='clusters')
    # ax=sns.barplot(data=clusteringDBSCAN,x='pyro2',y='count',hue='clustersDBSCAN',palette="Set2",ax=axes[0])
    # ax.bar_label(ax.containers[0],fontsize=10)
    # 
    # clusteringGMM=pd.merge(right=pd.concat([pd.DataFrame(dfXY.pyro2).reset_index(drop=True),pd.DataFrame(clustersGMM,columns=['clustersGMM'])],axis=1).drop_duplicates(),left=pd.DataFrame(dfXY.pyro2.value_counts()).reset_index(), how='inner',on='pyro2')
    # ax=sns.barplot(data=clusteringGMM,x='pyro2',y='count',hue='clustersGMM',palette="Set2",ax=axes[1],width=.2)#,ax=ax[1]
    # ax.bar_label(ax.containers[0],fontsize=10)
    # fig.tight_layout(pad=3)
    
    # In[37]:
    
    
    modelGMM = GaussianMixture(n_components=2).fit(dfXY.pyro2.to_numpy().reshape(-1,1))
    clustersGMM = modelGMM.predict(dfXY.pyro2.to_numpy().reshape(-1,1))
    #print(np.unique(clustersGMM))
    
    fig,ax = plt.subplots(1,1,figsize=(10,4))
    fig.suptitle("Clustering of pyro2goals with GMM",fontsize=20)
    #Compute df
    clusteringGMM=pd.merge(right=pd.concat([pd.DataFrame(dfXY.pyro2).reset_index(drop=True),pd.DataFrame(clustersGMM,columns=['clustersGMM'])],axis=1).drop_duplicates(),left=pd.DataFrame(dfXY.pyro2.value_counts()).reset_index(), how='inner',on='pyro2').sort_values(by=['pyro2'])
    #Plor barplot
    sns.barplot(data=clusteringGMM,x='pyro2',y='count',hue='clustersGMM',palette="Set2",width=.2,ax=ax)#,ax=ax[1]
    #Add labels
    barRectangles = ax.patches
    labels = clusteringGMM['count'].values
    label_i=0
    #Cannot loop through zip(barRectangles,labels) bc barRect is larger as there are rect in the bins with 0 counts
    for rect in barRectangles:#zip()stop when the shortest list ends, iterate through rect bars
        height = rect.get_height()
        if height != 0:
            #print(rect.get_x(),height+6,label)
            ax.text(x=rect.get_x()+rect.get_width()/2,y=height+6,s=str(int(height)),ha="center",va="bottom",color="black",fontsize=10)#s=str(labels[label_i])
            label_i+=1
    
    #Make room for all labels in plot
    ax.set_ylim(top=max(labels)+1500)
    #Other way, but not centered horizontally as u dont have the width of the bars
    #for index, value in dfXY.pyro2.value_counts().sort_index().items():
    #    axes[r,c].text(x=index, y=value, s=value, color='yellow', ha='center')
    
    fig.tight_layout(pad=3)
    
    
    # In[39]:
    
    
    #Assign meaningfull labels to clusters
    a=pd.DataFrame(modelGMM.means_,columns=["center"]).assign(label = lambda x: x.iloc[:,0].map(lambda y: "HighGoal" if y==x.max()[0] else "LowGoal")).reset_index(names="cluster")
    a
    
    
    # In[40]:
    
    
    #Print pyro2 clusters with new label
    fig,ax = plt.subplots(1,1,figsize=(10,4))
    fig.suptitle("Clustering of pyro2goals with GMM",fontsize=20)
    #Compute df
    clusteringGMM=clusteringGMM.assign(label=lambda x: x.clustersGMM.map(lambda y: "HighGoal" if y==a.loc[a.label=="HighGoal","cluster"].values[0] else "LowGoal").astype("category"))
    #Plor barplot
    sns.barplot(data=clusteringGMM,x='pyro2',y='count',hue='label',palette="Set2",width=.2,ax=ax)#,ax=ax[1]
    #Add labels
    barRectangles = ax.patches
    labels = clusteringGMM['count'].values
    label_i=0
    #Cannot loop through zip(barRectangles,labels) bc barRect is larger as there are rect in the bins with 0 counts
    for rect in barRectangles:#zip()stop when the shortest list ends, iterate through rect bars
        height = rect.get_height()
        if height != 0:
            #print(rect.get_x(),height+6,label)
            ax.text(x=rect.get_x()+rect.get_width()/2,y=height+6,s=str(int(height)),ha="center",va="bottom",color="black",fontsize=10)#s=str(labels[label_i])
            label_i+=1
    
    #Make room for all labels in plot
    ax.set_ylim(top=max(labels)+1500)
    fig.tight_layout(pad=3)
    
    
    # In[41]:
    
    
    #Add clusters to dfXY
    #if dfXY.SteelFamCluster.shape: dfXY.drop("SteelFamCluster",axis=1,inplace=True)
    dfXY.loc[:,'SteelFamCluster']=pd.DataFrame(clustersGMM,columns=['clustersGMM'],index=dfXY.index,dtype=np.int64).map(lambda y: "HighGoal" if y==a.loc[a.label=="HighGoal","cluster"].values[0] else "LowGoal").astype("category")
    dfXY=dfXY.copy()
    
    
    # In[42]:
    #Statistic description of numeric vars
    dfXY_description=dfXY.describe()

    
    
    # In[43]:
    #Cuts to consider outliers
    cut_neg = dfXY_description.iloc[1] - 3*dfXY_description.iloc[2] #mean-3*std
    cut_pos = dfXY_description.iloc[1] + 3*dfXY_description.iloc[2] #mean+3*std
    cut_neg.name = 'cut-'
    cut_pos.name = 'cut+'
    cut_neg = pd.DataFrame(cut_neg).T
    cut_pos = pd.DataFrame(cut_pos).T
    tbl_desc = pd.concat([dfXY_description,cut_neg, cut_pos])
    
    
    # In[44]:
    #Final Tune cut
    for c in range(tbl_desc.shape[1]):
        if tbl_desc.loc['cut-',tbl_desc.columns[c]]<0:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
        if tbl_desc.columns[c]=='CurrentWidth':
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0 #some widthIDs are smaller than the actual cut-
        elif tbl_desc.columns[c]=='NetWeight':
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=5000 #min around 20k, if cut for defects would be 10k. To be conservative 5k
        elif tbl_desc.columns[c] in tbl_desc.columns[tbl_desc.columns.str.contains('Ratio',regex=False)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
            tbl_desc.loc['cut+',tbl_desc.columns[c]]=4
        elif tbl_desc.columns[c] in tbl_desc.columns[tbl_desc.columns.str.contains('.*ControlValve.*|.*CV.*',regex=True)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
            tbl_desc.loc['cut+',tbl_desc.columns[c]]=100
        elif tbl_desc.columns[c] in tbl_desc.columns[tbl_desc.columns.str.contains('.*Zone0.*',regex=True) & ~tbl_desc.columns.str.contains('.*Zone0.*Ratio.*',regex=True) & ~tbl_desc.columns.str.contains('.*Temp.*',regex=True)]:#Zone0 is in %
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=0
            tbl_desc.loc['cut+',tbl_desc.columns[c]]=100
        elif tbl_desc.columns[c] in zoneTemp_vars_series[~zoneTemp_vars_series.str.contains("CV",regex=False)]:
            if tbl_desc.loc['cut-',tbl_desc.columns[c]]<=500:
                tbl_desc.loc['cut-',tbl_desc.columns[c]]=500 #Zone 0 seems to have many values near 0 that are outside the bell
        elif tbl_desc.columns[c] in pyro_vars_series[pyro_vars_series.str.contains("Temp",regex=False)]:
            tbl_desc.loc['cut-',tbl_desc.columns[c]]=1400 #pyro sensors seem to have many values at 1200 that are outside the bimodal bell, p() bc it is the lower limit of the sensor
    
    return dfXY
    
