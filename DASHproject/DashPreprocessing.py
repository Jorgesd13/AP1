from datetime import date

def func_preprocess_data(df):
    ###########IMPORT LIBRARIES###############
    import pandas as pd
    import numpy as np
    import os

    dfXY = df.copy()
    ########Make Air/Gas vars in 1-100%############
    #Zone1
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/59655*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/5170*100
    ##########################################################
    #Zone2
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/59655*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/5170*100
    numerical_var_series=dfXY[dfXY.columns[dfXY.dtypes == np.float64]].columns#.to_list()#[0:5]
    zoneAirGas_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Air.*|.*Zone.*Gas.*",regex=True)]].columns#.head(1)
    #Zone2 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
    dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]=dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]*47565/59655
    ###########################################################
    #Zone3
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/52831*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/3350*100
    #############################################################
    #Zone4
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/51951*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/3685*100
    #############################################################
    #Zone5
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/40318*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/2860*100
    ##############################################################
    #Zone6
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/41004*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/2860*100
    ###############################################################
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
    ########################################################
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
    
    return dfXY.copy()








def func_import_data(start_date:str,days:int=5):

    ###########IMPORT LIBRARIES###############
    from DigiPythonTools import DataUtility
    import pandas as pd
    import numpy as np
    import os
    import yaml
    import pyodbc
    from sklearn.mixture import GaussianMixture

    #############IMPORT DATA###############################
    ###Query table TAGS####
    #print(days)
    query=f'''
     DECLARE @start_from_date AS DATETIME =	(SELECT 
											IIF('{start_date}'<MIN(ts), MIN(ts),'{start_date}') AS min_date
										FROM db_name.schema_name.tblAp1FceTagHistory);
										
    DECLARE @end_from_date AS DATETIME =	(SELECT 
											IIF(DATEADD(DAY, {days}*(+1), @start_from_date)>MAX(ts), MAX(ts),DATEADD(DAY, {days}*(+1), @start_from_date)) AS max_date
										FROM db_name.schema_name.tblAp1FceTagHistory););
										
    select *
    from db_name.schema_name.tblAp1FceTagHistory);
    where ts between @start_from_date and @end_from_date --@MaxTime, '2024-08-11'
    order by ts asc'''#.format(d=days,start_date=start_date)
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
    from  db_name.schema_name.tblProceso tp
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
    FROM  db_name.schema_name.[Pyro2goals] p2g
    INNER JOIN  db_name.schema_name.catSteelGrades csg ON CAST(p2g.grade AS VARCHAR(4))=csg.Name
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
    ########Make Air/Gas vars in 1-100%############
    #Zone1
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomAirFlowFeedback.*|.*Zone1TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/59655*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone1BottomGasFlow.*|.*Zone1TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/5170*100
    ##########################################################
    #Zone2
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomAirFlowFeedback.*|.*Zone2TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/59655*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone2BottomGasFlow.*|.*Zone2TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/5170*100
    numerical_var_series=dfXY[dfXY.columns[dfXY.dtypes == np.float64]].columns#.to_list()#[0:5]
    zoneAirGas_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Air.*|.*Zone.*Gas.*",regex=True)]].columns#.head(1)
    #Zone2 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
    dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]=dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]*47565/59655
    ###########################################################
    #Zone3
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomAirFlowFeedback.*|.*Zone3TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/52831*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone3BottomGasFlow.*|.*Zone3TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/3350*100
    #############################################################
    #Zone4
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomAirFlowFeedback.*|.*Zone4TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/51951*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone4BottomGasFlow.*|.*Zone4TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/3685*100
    #############################################################
    #Zone5
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomAirFlowFeedback.*|.*Zone5TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/40318*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone5BottomGasFlow.*|.*Zone5TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/2860*100
    ##############################################################
    #Zone6
    #Air
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomAirFlowFeedback.*|.*Zone6TopAirFlowFeedback.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/41004*100
    #Gas
    #dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]].head(2)
    dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]] = dfXY.loc[:,dfXY.columns[(dfXY.columns.str.contains(".*Zone6BottomGasFlow.*|.*Zone6TopGasFlow.*",regex=True)) & (~dfXY.columns.str.contains("ControlValve",regex=False))]]/2860*100
    ###############################################################
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
    ########################################################
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

    ###CREATE SteelFamCluster VARS ###
    #Compute Clusters
    #print(dfXY.shape)
    #print(dfXY.pyro2.to_numpy().reshape(-1,1).shape)
    modelGMM = GaussianMixture(n_components=2).fit(dfXY.pyro2.to_numpy().reshape(-1,1))
    #print("Predict clusters")
    clustersGMM = modelGMM.predict(dfXY.pyro2.to_numpy().reshape(-1,1))
    #print("Merge")
    #Assign meaningfull labels to clusters
    dfClusterLabels=pd.DataFrame(modelGMM.means_,columns=["center"]).assign(label = lambda x: x.iloc[:,0].map(lambda y: "HighGoal" if y==x.max()[0] else "LowGoal")).reset_index(names="cluster")
    clusteringGMM=pd.merge(right=pd.concat([pd.DataFrame(dfXY.pyro2).reset_index(drop=True),pd.DataFrame(clustersGMM,columns=['clustersGMM'])],axis=1).drop_duplicates(),left=pd.DataFrame(dfXY.pyro2.value_counts()).reset_index(), how='inner',on='pyro2').sort_values(by=['pyro2'])
    #Compute df
    clusteringGMM=clusteringGMM.assign(label=lambda x: x.clustersGMM.map(lambda y: "HighGoal" if y==dfClusterLabels.loc[dfClusterLabels.label=="HighGoal","cluster"].values[0] else "LowGoal").astype("category"))
    #Add clusters to dfXY
    dfXY.loc[:,'SteelFamLabel']=pd.DataFrame(clustersGMM,columns=['clustersGMM'],index=dfXY.index,dtype=np.int64).map(lambda y: "HighGoal" if y==dfClusterLabels.loc[dfClusterLabels.label=="HighGoal","cluster"].values[0] else "LowGoal").astype("category")
    dfXY.loc[:,'SteelFamCluster']=dfXY.loc[:,'SteelFamLabel'].map(lambda x: 1 if x=="HighGoal" else 0).astype(np.int64)

    return dfXY;


# df=func_import_data('2024-08-05',5)
# print(df.shape)
# print(df.index[0])

