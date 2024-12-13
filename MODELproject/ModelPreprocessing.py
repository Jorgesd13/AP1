
def func_import_data(days:int=30):

    ###########IMPORT LIBRARIES###############
    from DigiPythonTools import DataUtility
    import pandas as pd
    import numpy as np
    import os
    import yaml
    import pyodbc
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.feature_selection import VarianceThreshold

    #############IMPORT DATA###############################
    ###Query table TAGS####
    #print(days)
    query='''
    DECLARE @MaxTime AS DATETIME = (SELECT MAX(ts) FROM Digitalization.ApEng.tblAp1FceTagHistory);
    
    select *
    from Digitalization.ApEng.tblAp1FceTagHistory
    where ts between DATEADD(DAY,-{d},@MaxTime) and @MaxTime --@MaxTime, '2024-08-11'
    order by ts asc'''.format(d=days)
    #os.chdir(r"C:\Users\js5296\PycharmProjects\ECP2_AP1")
    min_sql = DataUtility()
    dfTags = pd.DataFrame()
    dfTags = min_sql.min_query(query)
    #os.chdir(r"C:\Users\js5296\PycharmProjects\ECP2_AP1\Dash")
    #print("Shape dfTags",dfTags.shape)
    ###PIVOT FLOAT VARS dfTags####
    dfFloatTags = dfTags[~dfTags.Tagname.isin(["AP1_FurnaceCoilID"])].pivot(index='ts',columns='Tagname',values='Value')

    ###PIVOT STRING VARS dfTags###
    dfStringTags=dfTags[dfTags.Tagname.isin(["AP1_FurnaceCoilID"])].pivot(index="ts",columns="Tagname",values="vValue")

    ###Merge float and string tbls###
    dfHistory = pd.merge(left=dfStringTags,right=dfFloatTags,how="inner",left_index=True,right_index=True)
    dfHistory.AP1_FurnaceCoilID=dfHistory.AP1_FurnaceCoilID.str.upper()
    #print("Shape dfHistory",dfHistory.shape)

    ###QUERY PROCESO###
    coils = dfHistory['AP1_FurnaceCoilID'].str.upper().unique().tolist()
    real_coils = []
    for coil in coils:
        if type(coil) == str:
            real_coils.append(coil)
    real_coils = "'" + "','".join(real_coils)  + "'"

    query2  = '''select top 2000000 LastProcessDate, CoilNumber, ProductID, ProductDivision, CurrentGuage, CurrentWidth, CoilLength, NetWeight, SteelGradeID
    from nasmes.AS400.tblProceso tp
    where  (LineID = 25) and ((tp.ProductID + COALESCE(tp.ProductDivision,'')) in ({coil_list})) order by LastProcessDate'''.format(coil_list = real_coils)

    min_sql = DataUtility()
    dfProceso = pd.DataFrame()
    dfProceso = min_sql.min_query(query2)

    dfProceso.set_index("LastProcessDate",inplace=True)
    dfProceso.CoilNumber=dfProceso.CoilNumber.str.upper()
    dfProceso.ProductID=dfProceso.ProductID.str.upper()
    dfProceso.ProductDivision=dfProceso.ProductDivision.str.upper()
    #print("Shape dfProceso",dfProceso.shape)

    ###MERGE TAGS AND PROCESO###
    dfXLong = pd.merge_asof(right=dfProceso,left=dfHistory,right_by="CoilNumber",left_by="AP1_FurnaceCoilID",right_index=True,left_index=True,direction="nearest",tolerance=pd.Timedelta("1D"))#
    #print("Shape dfXLong",dfXLong.shape)
    ###CLEAN COIL MISMATCHES DUE TO TRANSITION COILS###
    dfXMismatch = dfXLong[dfXLong.AP1_FurnaceCoilID != dfXLong.CoilNumber]
    mismatchedCoilsX=list(dfXMismatch.AP1_FurnaceCoilID.unique())
    dfX=dfXLong[~dfXLong.AP1_FurnaceCoilID.isin(mismatchedCoilsX)]
    #print("Shape dfX",dfX.shape)

    ###QUERY GOALS###
    query3 = '''SELECT TOP (3000) [grade]
        ,[pyro2]
        ,[line]
        ,[Special]
        ,csg.SteelGradeID
    FROM [Digitalization].[QCEng].[Pyro2goals] p2g
    INNER JOIN NASMES.NAS.catSteelGrades csg ON CAST(p2g.grade AS VARCHAR(4))=csg.Name
    WHERE line = 3310'''

    min_sql = DataUtility()
    dfY = pd.DataFrame()
    dfY = min_sql.min_query(query3)
    #Drop 'line' and 'Special' from dfY
    dfY.drop(["line","Special"],axis=1,inplace=True)
    #Set SteelGradeID in dfX and dfY of same type
    dfX=dfX.astype({'SteelGradeID': str(dfY.SteelGradeID.dtypes)})
    #print("Shape dfY",dfY.shape)

    ###MERGE X and Y###
    dfXY = dfX.reset_index().merge(right=dfY,how="left",on="SteelGradeID").set_index("ts")
    #print("Shape dfXY",dfXY.shape)

    ###VARIABLE IMPORT CLEANUP###
    #MismTches between SteelGradeID and Goals
    mismatchedSteelGradeID = [item for item in dfX.SteelGradeID.sort_values().unique() if item not in dfY.SteelGradeID.sort_values().unique()]
    #Set new goal to SteelGradeIDs 1,2,4:
    if dfXY.SteelGradeID.isin([1,2,4]).sum() !=0:
        dfXY.loc[dfXY[dfXY.SteelGradeID.isin([1,2,4])].index,["grade","pyro2"]]=dfXY[dfXY.SteelGradeID.isin([3,34])][["grade","pyro2"]].iloc[0].to_list()
    #Drop SteelGradeIDs 97,109,238,290 (mismatchedSteelGradeIDs that are not 1,2,4)
    mismatchedSteelGradeIDNew = [item for item in mismatchedSteelGradeID if item not in [1,2,4]]
    dfXY = dfXY[~dfXY.SteelGradeID.isin([97,109,238,290])]
    #Strip column names
    dfXY.columns=dfXY.columns.str.strip(' \t\n')

    ###Remove Unused Vars###
    dfXY_coils = dfXY.AP1_FurnaceCoilID  # we keep the indexed coil numbers in case
    dfXY.drop(["AP1_FurnaceCoilID", "CoilNumber", "ProductID", "ProductDivision", "grade"], axis=1, inplace=True)
    if "SteelGradeID" in dfXY.columns:
        dfXY.drop("SteelGradeID",axis=1,inplace=True)
    if "AP1_FurnaceRWPEntryTemp" in dfXY.columns:
        dfXY.drop("AP1_FurnaceRWPEntryTemp",axis=1,inplace=True)
    if "AP1_FurnaceRWPExitTemp" in dfXY.columns:
        dfXY.drop("AP1_FurnaceRWPExitTemp",axis=1,inplace=True)
    ###Create Numeric Vars###
    numerical_var_series = dfXY[dfXY.columns[dfXY.dtypes == np.float64]].columns
    # Create the different groups of columns to plot
    proceso_vars_series = dfXY[numerical_var_series[(~numerical_var_series.str.contains("AP1_Furnace", regex=False)) & (~numerical_var_series.str.contains("pyro2", regex=False))]].columns  # .head(1)
    zoneTemp_vars_series = dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Temp.*", regex=True)]].columns  # .head(1)
    zoneAirGas_vars_series = dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Air.*|.*Zone.*Gas.*", regex=True)]].columns  # .head(1)
    recuperator_vars_series = dfXY[numerical_var_series[numerical_var_series.str.contains("AP1_Furnace.*Dilution|Recuperator|RWP|Combustion.*",regex=True)]].columns  # .head(1)
    pyro_vars_series = dfXY[numerical_var_series[numerical_var_series.str.contains(".*pyro.*|.*Pyro.*", regex=True)]].columns  # .head(1)
    rest_vars_series = dfXY[[var for var in numerical_var_series.to_list() if var not in proceso_vars_series.to_list() and var not in zoneTemp_vars_series.to_list() and var not in zoneAirGas_vars_series.to_list() and var not in recuperator_vars_series.to_list() and var not in pyro_vars_series.to_list()]].columns  # .head(1)
    ###Rescale Outdated vars###
    # Zone1 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
    dfXY.loc[:, zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone1.*AirFlowSet[Pp]oint.*", regex=True)]] = dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone1.*AirFlowSet[Pp]oint.*",regex=True)]] * 47565 / 59655
    # Zone2 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
    dfXY.loc[:, zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*", regex=True)]] = dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]] * 47565 / 59655

    ###REMOVE OUTLIERS###
    dfXY_description = dfXY.describe()
    # Cuts to consider outliers
    cut_neg = dfXY_description.iloc[1] - 3 * dfXY_description.iloc[2]  # mean-3*std
    cut_pos = dfXY_description.iloc[1] + 3 * dfXY_description.iloc[2]  # mean+3*std
    cut_neg.name = 'cut-'
    cut_pos.name = 'cut+'
    cut_neg = pd.DataFrame(cut_neg).T
    cut_pos = pd.DataFrame(cut_pos).T
    tbl_desc = pd.concat([dfXY_description, cut_neg, cut_pos])
    # Tune cut
    for c in range(tbl_desc.shape[1]):
        if tbl_desc.loc['cut-', tbl_desc.columns[c]] < 0:
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 0
        if tbl_desc.columns[c] == 'AP1_FurnaceWidth':
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 0  # some widthIDs are smaller than the actual cut-
        elif tbl_desc.columns[c] == 'NetWeight':
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 5000  # min around 20k, if cut for defects would be 10k. To be conservative 5k
        elif tbl_desc.columns[c] in tbl_desc.columns[tbl_desc.columns.str.contains('Ratio', regex=False)]:
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 0
            tbl_desc.loc['cut+', tbl_desc.columns[c]] = 4
        elif tbl_desc.columns[c] in tbl_desc.columns[
            tbl_desc.columns.str.contains("P1_FurnaceZone.*[C].*[V].*", regex=True)]:
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 0
            tbl_desc.loc['cut+', tbl_desc.columns[c]] = 100  # CV vars are in %
        elif tbl_desc.columns[c] in zoneAirGas_vars_series[
            zoneAirGas_vars_series.str.contains("Zone0", regex=False) & ~zoneAirGas_vars_series.str.contains("Ratio",regex=False)]:
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 0
            tbl_desc.loc['cut+', tbl_desc.columns[c]] = 100  # Zone0 airgas vars are in %
        elif tbl_desc.columns[c] in zoneTemp_vars_series[~zoneTemp_vars_series.str.contains("CV", regex=False)]:
            if tbl_desc.loc['cut-', tbl_desc.columns[c]] <= 500:
                tbl_desc.loc['cut-', tbl_desc.columns[c]] = 500  # Zone 0 seems to have many values near 0 that are outside the bell
        elif tbl_desc.columns[c] in pyro_vars_series[pyro_vars_series.str.contains("Temp", regex=False)]:
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 1400  # pyro sensors seem to have many values at 1200 that are outside the bimodal bell, p() bc it is the lower limit of the sensor

    # Remove outliers
    for var in tbl_desc.columns:
        dfXY.loc[:, var] = dfXY.loc[:, var].map(lambda x: x if ((x >= tbl_desc.loc["cut-", var]) & (x <= tbl_desc.loc["cut+", var])) else np.nan,na_action='ignore')
    # Remove outliers when LineSpeed=0
    for t in dfXY.index:
        if (dfXY.loc[t, "AP1_FurnaceLineSpeed"] == 0) | (np.isnan(dfXY.loc[t, "AP1_FurnaceLineSpeed"])):
            dfXY.loc[t, dfXY.columns[~dfXY.columns.str.contains("AP1_FurnaceLineSpeed")]] = np.nan

    ###CREATE SteelFamCluster VARS###
    #Compute Clusters
    modelGMM = GaussianMixture(n_components=2).fit(dfXY.pyro2.dropna().to_numpy().reshape(-1,1))
    clustersGMM = modelGMM.predict(dfXY.pyro2.dropna().to_numpy().reshape(-1,1))
    #Assign meaningfull labels to clusters
    clusteringGMM_Long = pd.DataFrame(dfXY.pyro2, columns=['pyro2'])  # .reset_index(drop=True)
    c=0
    for t in dfXY.index:
        if np.isnan(clusteringGMM_Long.loc[t, "pyro2"]):
            clusteringGMM_Long.loc[t, "SteelFamCluster"] = np.nan
        else:
            clusteringGMM_Long.loc[t, "SteelFamCluster"] = clustersGMM[c]
            c += 1
    clusteringGMM = pd.merge(right=clusteringGMM_Long.drop_duplicates().reset_index(drop=True),
                             left=pd.DataFrame(dfXY.pyro2.value_counts()).reset_index(), how='inner',on='pyro2').sort_values(by=['pyro2'])
    dfClusterLabels = pd.DataFrame(modelGMM.means_, columns=["center"]).assign(
        label=lambda x: x.iloc[:, 0].map(lambda y: "HighGoal" if y == x.max()[0] else "LowGoal")).reset_index(names="cluster")

    # Check mapping works before applying to df
    # Defining all the conditions for labelling
    def condition(y):
        if y == dfClusterLabels.loc[dfClusterLabels.label == "HighGoal", "cluster"].values[0]:
            return "HighGoal"
        elif np.isnan(y):
            return np.nan
        elif y == dfClusterLabels.loc[dfClusterLabels.label == "LowGoal", "cluster"].values[0]:
            return "LowGoal"
        else:
            pass

    clusteringGMM_Long=clusteringGMM_Long.assign(labelsGMM=lambda x: x.SteelFamCluster.map(condition).astype("category"))#.drop_duplicates()
    clusteringGMM = clusteringGMM.assign(labelsGMM=lambda x: x.SteelFamCluster.map(condition).astype("category"))
    #Add clusters to dfXY
    dfXY.loc[:,'SteelFamLabel']=clusteringGMM_Long.labelsGMM.astype("category")
    dfXY.loc[:,'SteelFamCluster']=clusteringGMM_Long.SteelFamCluster#.astype(np.int64)

    ###CREATE CSdeltaTmin###
    # CSdeltaT:Cross Section Time in Fce
    # iterate timestamps of dfXY, not dfXY_coils
    v0 = 0
    time = dfXY.index[0]
    dfXY['CSdeltaTmin'] = np.nan
    frow0 = 0
    for i, t in enumerate(dfXY.index):  # compute CS t of coil in Fce
        if (i == 0) & (dfXY.AP1_FurnaceLineSpeed.loc[t] == 0):  # First row:
            frow0 = 1
        if (dfXY.AP1_FurnaceLineSpeed.loc[t] == 0) & (frow0 == 1):
            dfXY.loc[t, 'CSdeltaTmin'] = np.nan  # dont know how much time will remain v=0
        elif (dfXY.AP1_FurnaceLineSpeed.loc[t] != 0) & (frow0 == 1):
            frow0 = 0
        if (dfXY.AP1_FurnaceLineSpeed.loc[t] == 0) & (v0 == 0) & (frow0 != 1):  # CS t inf
            v0 = 1
            time = t  # update timeindex when v=0
        elif (v0 == 1) & (dfXY.AP1_FurnaceLineSpeed.loc[t] != 0):
            dfXY.loc[time:dfXY.index[i - 1], 'CSdeltaTmin'] = (dfXY.index[i] - time).total_seconds() / 60  # if (dfXY.index[i-1]-time).total_seconds()/60 when v=0 only 1row, CSdeltaTmin=0.0
            dfXY.loc[t, 'CSdeltaTmin'] = 160 / dfXY.loc[t, 'AP1_FurnaceLineSpeed']
            v0 = 0
        elif (v0 == 0) & (dfXY.AP1_FurnaceLineSpeed.loc[t] != 0):
            dfXY.loc[t, 'CSdeltaTmin'] = 160 / dfXY.loc[t, 'AP1_FurnaceLineSpeed']
        if i == dfXY.shape[0] - 1:  # Last row
            if (v0 == 1) & (dfXY.AP1_FurnaceLineSpeed.loc[t] == 0):
                dfXY.loc[t, 'CSdeltaTmin'] = np.nan  # dont know how much time will remain v=0
    # dfXY.CSdeltaTmin
    dfXY = dfXY.copy()

    ###FINAL OUTLIER DETECTION FOR CSdeltaTmin###
    # Remove outliers when LineSpeed=0
    for t in dfXY.index:
        if (dfXY.loc[t, "AP1_FurnaceLineSpeed"] == 0) | (np.isnan(dfXY.loc[t, "AP1_FurnaceLineSpeed"])):
            dfXY.loc[t, dfXY.columns[~dfXY.columns.str.contains("AP1_FurnaceLineSpeed")]] = np.nan
    CSdeltaTmin_describe = pd.DataFrame(dfXY.CSdeltaTmin.describe())
    CSdeltaTmin_describe.loc["cut-", "CSdeltaTmin"] = CSdeltaTmin_describe.loc["mean", "CSdeltaTmin"] - 3 * CSdeltaTmin_describe.loc["std", "CSdeltaTmin"]
    CSdeltaTmin_describe.loc["cut+", "CSdeltaTmin"] = CSdeltaTmin_describe.loc["mean", "CSdeltaTmin"] + 3 * CSdeltaTmin_describe.loc["std", "CSdeltaTmin"]
    CSdeltaTmin_describe
    tbl_desc.loc[:, "CSdeltaTmin"] = CSdeltaTmin_describe
    # Tune cut
    for c in range(tbl_desc.shape[1]):
        if tbl_desc.loc['cut-', tbl_desc.columns[c]] < 0:
            tbl_desc.loc['cut-', tbl_desc.columns[c]] = 0
    # Remove outliers from CSdeltaTmin
    dfXY.loc[:, "CSdeltaTmin"] = dfXY.loc[:, "CSdeltaTmin"].map(lambda x: x if ((x >= tbl_desc.loc["cut-", "CSdeltaTmin"]) & (x <= tbl_desc.loc["cut+", "CSdeltaTmin"])) else np.nan,na_action='ignore')

    ###ENCODING CATEGORICAL DATA###
    # Make SteelFamCluster OrdinalEncoder:
    dfXY.SteelFamCluster = OrdinalEncoder().fit_transform(dfXY[["SteelFamCluster"]])
    print("Feature Selection")
    ###FEATURE SELECTION###
    # Create dfX & dfY#
    dfX = dfXY[proceso_vars_series.tolist() + zoneTemp_vars_series[zoneTemp_vars_series.str.contains(".*Fdbk.*|.*TopTemp.*", regex=True)].tolist() +
               zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Top.*F.*db.*k.*", regex=True)].tolist() +
               recuperator_vars_series[~recuperator_vars_series.str.contains(".*CV.*|.*SetPoint.*", regex=True)].tolist() + [pyro_vars_series[0]] +
               VarianceThreshold(threshold=0.0).set_output(transform="pandas").fit(dfXY[rest_vars_series]).get_feature_names_out().tolist() +
               ['CSdeltaTmin', 'SteelFamCluster','AP1_FurnacePyro2Temp']]
    dfX.iloc[:, :-1] = dfX.iloc[:, :-1].interpolate(method='linear', axis=0,limit=5)  # .isna().sum()#AP1_FurnaceZone0MasterTempFdbk has around 8k nan, whereas the rest of vars have aroun 1.1k only, after interpolate
    dfX = dfX.drop('AP1_FurnaceZone0MasterTempFdbk', axis=1)
    dfX = dfX.dropna()
    dfY = dfX[["AP1_FurnacePyro2Temp", "AP1_FurnacePyro1Temp"]]
    dfX = dfX.drop(['AP1_FurnacePyro2Temp', "AP1_FurnacePyro1Temp"], axis=1)
    # CREATE dfGroups#
    dfGroups = pd.DataFrame(dfXY_coils.loc[dfXY.index]).assign(group=lambda x: OrdinalEncoder().set_output(transform="pandas").fit_transform(x[["AP1_FurnaceCoilID"]]).astype(np.int64))
    # Check any variance 0 among features:
    dfX = dfX[VarianceThreshold(threshold=0.0).set_output(transform="pandas").fit(dfX).get_feature_names_out().tolist()]
    # Filter nan in dfGroups
    dfGroups = dfGroups.loc[dfX.index, :]

    # Create Temp Avg
    TopTemp1_2 = dfX.loc[:, dfX.columns.str.contains(".*1TopTemp.*|.*2TopTemp.*", regex=True)].mean(axis=1)
    TopTemp3_4_5 = dfX.loc[:, dfX.columns.str.contains(".*3TopTemp.*|.*4TopTemp.*|.*5TopTemp.*", regex=True)].mean(axis=1)
    TopTemp6_7_8 = dfX.loc[:, dfX.columns.str.contains(".*6TopTemp.*|.*7TopTemp.*|.*8TopTemp.*", regex=True)].mean(axis=1)
    # Create GasFlow Avg
    TopGas1_2 = dfX.loc[:, dfX.columns.str.contains(".*1TopGasFlow.*|.*2TopGasFlow.*", regex=True)].mean(axis=1)
    TopGas3_4_5 = dfX.loc[:,dfX.columns.str.contains(".*3TopGasFlow.*|.*4TopGasFlow.*|.*5TopGasFlow.*", regex=True)].mean(axis=1)
    TopGas6_7_8 = dfX.loc[:,dfX.columns.str.contains(".*6TopGasFlow.*|.*7TopGasFlow.*|.*8TopGasFlow.*", regex=True)].mean(axis=1)
    #Frop Air, Gas, Temp vars
    dfX.drop(dfX.columns[dfX.columns.str.contains("TopTemp", regex=False)].tolist(), axis=1, inplace=True)
    dfX.drop(dfX.columns[dfX.columns.str.contains("TopGasFlow", regex=False)].tolist(), axis=1, inplace=True)
    dfX.drop(dfX.columns[dfX.columns.str.contains("TopAirFlow", regex=False)].tolist(), axis=1, inplace=True)
    #Add Temp avg
    dfX.loc[:, 'TopTemp1_2'] = TopTemp1_2
    dfX.loc[:, 'TopTemp3_4_5'] = TopTemp3_4_5
    dfX.loc[:, 'TopTemp6_7_8'] = TopTemp6_7_8
    #Add Gas avg
    dfX.loc[:, 'TopGas1_2'] = TopGas1_2
    dfX.loc[:, 'TopGas3_4_5'] = TopGas3_4_5
    dfX.loc[:, 'TopGas6_7_8'] = TopGas6_7_8

    dfX = dfX[dfX.columns[(~dfX.columns.str.contains("SteelFamCluster", regex=False))].tolist() + ["SteelFamCluster"]]


    return dfX, dfY, dfGroups;


# dfXY, dfX, dfY, dfGroups = func_import_data(30)
# print(dfXY.shape)
# print(dfX.shape)
# print(dfXY.index[0])
# print(dfX.index[0])
# print(dfX.columns.tolist())

