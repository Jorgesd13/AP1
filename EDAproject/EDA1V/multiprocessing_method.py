import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import PercentFormatter
from scipy import stats as st
from scipy.stats import norm, beta
# !pip install --no-input kaleido==0.2.1
import kaleido #0.2.1
# !pip uninstall --no-input plotly
# !pip install --no-input plotly==5.5.0
import plotly
import plotly.io as pio #plotly 5.5.0
import plotly.graph_objects as go
import plotly_express as px
from plotly.subplots import make_subplots
import gc
import asyncio
from async_timeout import timeout
#from asgiref.sync import sync_to_async
import time,datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal
import os
import pickle
from EDA1V_new import func_preprocessing


dfXY = func_preprocessing()


#Numeric vars
numerical_var_series=dfXY[dfXY.columns[dfXY.dtypes == np.float64]].columns#.to_list()#[0:5]
#print(type(numerical_var_series))
#print(len(numerical_var_series))
numerical_var_series




#Create the different groups of columns to plot
proceso_vars_series=dfXY[numerical_var_series[(~numerical_var_series.str.contains("AP1_Furnace",regex=False)) & (~numerical_var_series.str.contains("pyro2",regex=False))]].columns#.head(1)
zoneTemp_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Temp.*",regex=True)]].columns#.head(1)

zoneAirGas_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Air.*|.*Zone.*Gas.*",regex=True)]].columns#.head(1)

recuperator_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains("AP1_Furnace.*Dilution|Recuperator|RWP|Combustion.*",regex=True)]].columns#.head(1)

pyro_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*pyro.*|.*Pyro.*",regex=True)]].columns#.head(1)

rest_vars_series=dfXY[[var for var in numerical_var_series.to_list() if var not in proceso_vars_series.to_list() and var not in zoneTemp_vars_series.to_list() and var not in zoneAirGas_vars_series.to_list() and var not in recuperator_vars_series.to_list() and var not in pyro_vars_series.to_list()]].columns#.head(1)





#Zone2 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]=dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]*47565/59655



gasFdbk_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*GasFlowF.*db.*k.*",regex=True)]
gasFdbk_vars_series.name="gasFdbk_vars_series"
airFdbk_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*AirFlowF.*db.*k.*",regex=True)]
airFdbk_vars_series.name="airFdbk_vars_series"
airSP_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*AirFlow[Ss]{1}.*[Pp]{1}.*",regex=True)]
airSP_vars_series.name="airSP_vars_series"
gasCV_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*GasFlowC.*V.*",regex=True)]
gasCV_vars_series.name="gasCV_vars_series"
airCV_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*AirFlowC.*V.*",regex=True)]
airCV_vars_series.name="airCV_vars_series"



#Define function plot scatters plotly downsamplings
def func_downsample_scatter_plotly_AirGas(xlist_series:pd.Series,ylist_series:pd.Series):#async 
    try:
        print("entered the function")
        dfDown=dfXY.iloc[[int(x) for x in list(np.round(np.linspace(0,dfXY.shape[0]-1,2000),0))],:]#Downsample evenly
        ncol=9
        nrow=2
        var=0
        subtitles_list=[]
        for i in range(2):#Bottom/Top
            for j in range(9):#9zones
                if i==0:
                    subtitles_list.append(f"Zone{j} Top")
                else:
                    subtitles_list.append(f"Zone{j} Bottom")
                
        fig = make_subplots(rows=nrow,cols=ncol,subplot_titles=subtitles_list,vertical_spacing=0.1)#vertical_spacing(0-1)
    
        for r in range(1,nrow+1):
            if r==1:
                var=1
            else:
                var=0
            for c in range(1,ncol+1):
                #Create x,y,yhat
                x=np.asarray(dfDown[xlist_series[var]]).reshape(-1,1)
                y= np.asarray(dfDown[ylist_series[var]])
                y_pred = LinearRegression().fit(x,y).predict(x)
                x=x[:,0]
                #plot Scatters
                if (r==1) & (c==1):
                    fig.add_trace(go.Scatter(x=dfDown[dfDown.SteelFamCluster=="HighGoal"][xlist_series[var]], 
                            y=dfDown[dfDown.SteelFamCluster=="HighGoal"][ylist_series[var]], 
                            mode='markers', 
                            #zorder=3,
                             name="HighGoal",
                             legendgroup="group1",
                             marker=dict(color="blue",line_width=0.6)
                            #,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                ),row=r,col=c)
                    fig.add_trace(go.Scatter(x=dfDown[dfDown.SteelFamCluster=="LowGoal"][xlist_series[var]], 
                            y=dfDown[dfDown.SteelFamCluster=="LowGoal"][ylist_series[var]], 
                            mode='markers', 
                            #zorder=3,
                             name="LowGoal",
                             legendgroup="group2",
                             marker=dict(color="green",line_width=0.6)
                            #,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                ),row=r,col=c)
                else:
                    fig.add_trace(go.Scatter(x=dfDown[dfDown.SteelFamCluster=="HighGoal"][xlist_series[var]], 
                            y=dfDown[dfDown.SteelFamCluster=="HighGoal"][ylist_series[var]], 
                            mode='markers', 
                             #zorder=3,
                             name="HighGoal",
                             legendgroup="group1",
                             marker=dict(color="blue",line_width=0.6)
                            #,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                ,showlegend=False),row=r,col=c)
                    fig.add_trace(go.Scatter(x=dfDown[dfDown.SteelFamCluster=="LowGoal"][xlist_series[var]], 
                            y=dfDown[dfDown.SteelFamCluster=="LowGoal"][ylist_series[var]], 
                            mode='markers', 
                            #zorder=3,
                             name="LowGoal",
                             legendgroup="group2",
                             marker=dict(color="green",line_width=0.6)
                            #,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                ,showlegend=False),row=r,col=c)
        
                #plot regression line
                fig.add_trace(go.Scatter(x=x, 
                        y=y_pred, 
                        mode='lines', 
                        #zorder=4,
                        #name='2',
                        line=dict(width=1.5,color="orange"),
                        showlegend=False),row=r,col=c)
                #plot regression data
                fig.add_annotation(x=dfDown[xlist_series[var]].min(), y=dfDown[ylist_series[var]].max()+10,
                        #text="r\u00b2={:.2f}, p={:.2E}".format(st.pearsonr(x,y)[0]**2,st.pearsonr(x,y)[1]),
                       text="r\u00b2={:.2f}".format(st.pearsonr(x,y)[0]**2),
                       #hovertext="pepe",
                       align='center',
                       showarrow=False,
                       xanchor='left',
                       yanchor='bottom',
                       #clicktoshow="onout",
                       #captureevents=True,
                       #xref='x2',
                       #yref='y2',
                       row=r,
                       col=c,
                      bordercolor='black',
                      borderpad=4,
                       borderwidth=2,
                      bgcolor='white',
                      font=dict(color='black'))
                #Add xaxis labels
                #fig.update_xaxes(title_text=xlist_series[var], row=r, col=c)#range=[40, 80],showgrid=False,type="log",
                #Add yaxis labels
                #fig.update_yaxes(title_text=ylist_series[var], row=r, col=c)
                #update var to plot
                var+=2
    
    
    
                
        #Add title and size and show img
        fig.update_layout(height=900, width=1850,title_text=f'DownSample x={xlist_series.name} vs y={ylist_series.name} across all zones', showlegend=True,title_x=0.5)#height=500*9, width=1000, #, legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01)
        #fig.show()
        if not os.path.exists(r"C:\Users\carlo\Desktop\Projects\AP1\AP1_repo\EDAproject\images\1V\ScatterPlots\Dash"):
            os.mkdir(r"C:\Users\carlo\Desktop\Projects\AP1\AP1_repo\EDAproject\images\1V\ScatterPlots\Dash")
        # fig.write_html("C:\\Users\\js5296\\Desktop\\NAS\\Projects\\AP1_Furnace_Control\\EDAs\\old\\1V\\Dash\\DownSample_{}_vs_{}.html".format(ylist_series.name,xlist_series.name))
        # fig.write_image("C:\\Users\\js5296\\Desktop\\NAS\\Projects\\AP1_Furnace_Control\\EDAs\\old\\1V\\Dash\\DownSample_{}_vs_{}.webp".format(ylist_series.name,xlist_series.name))
        try:
            #print(os.getcwd())
            # Try saving the figure to both locations

            #time.sleep(60) #To make the multiprocessing func catch a time exception.
            fig.write_html("C:\\Users\\carlo\\Desktop\\Projects\\AP1\\AP1_repo\\EDAproject\\EDA1V\\DownSample_{}_vs_{}.html".format(ylist_series.name,xlist_series.name))
            #print("Saved figure to: DownSample_{}_vs_{}.html".format(ylist_series.name,xlist_series.name))
            
            fig.write_html("C:\\Users\\carlo\\Desktop\\Projects\\AP1\\AP1_repo\\EDAproject\\images\\1V\\ScatterPlots\\Dash\\newDownSample_{}_vs_{}.html".format(ylist_series.name,xlist_series.name))
            #print("Saved figure to: C:\\Users\\carlo\\Desktop\\Projects\\AP1\\AP1_repo\\EDAproject\\images\\1V\\ScatterPlots\\Dash\\newDownSample_{}_vs_{}.html".format(ylist_series.name,xlist_series.name))
            print("html save")
            
        except Exception as e:
            print(f"Error while saving figure: {e}")
            
        #fig.write_html("/doc/NB_dir/js5296/AP1/EDA1V/img/DownSample_{}_vs_{}.html".format(ylist_series.name,xlist_series.name))
        # print("webp save")#GIVES ERROR IN LOCAL ENV
        # fig.write_image("{}_vs_{}.png".format(ylist_series.name,xlist_series.name))
        # fig.write_image("{}_vs_{}.jpeg".format(ylist_series.name,xlist_series.name))
        # fig.write_image("{}_vs_{}.webp".format(ylist_series.name,xlist_series.name))
        # fig.write_image("/doc/NB_dir/js5296/AP1/EDA1V/img/DownSample_{}_vs_{}.webp".format(ylist_series.name,xlist_series.name))#For coldmill jupyter
    
        #fig.show()
        del fig
        gc.collect()
        #return fig.layout

    except Exception as e:
        print(f"Error during execution: {e}")
        raise 







import multiprocessing
from multiprocessing import Process

def func_print(x:int=2,y:int=4):
    time.sleep(1*60)
    print(x+y)
    
def func_run_with_multiprocessing(xlist_series:pd.Series,ylist_series:pd.Series, timeout_seconds:int=300): #,x:int, y:int
    # Create a process for the long-running function
    process = Process(target=func_downsample_scatter_plotly_AirGas, args=(xlist_series, ylist_series))
    #process = Process(target=func_print, args=(x, y))
    
    try:
        # Start the process
        print("Starting the process...")
        process.start()
        #process.run()
        print("Process started.")
        # Wait for the process to finish with a timeout
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            print("Process exceeded the timeout.")
            process.terminate()
            raise TimeoutError("The function exceeded the time limit!")
        else:
            print("Process didn´t exceed the timeout.")
    except TimeoutError:
        print("Caught timeout exception!")
    except Exception as e:
        # Catch any other exceptions that occur during the function execution
        print(f"Caught an error during function execution: {e}")
    else:
        print("Function completed successfully.")


#Example usage
#func_print(1,7)
if __name__ == "__main__":
    func_run_with_multiprocessing(xlist_series=gasCV_vars_series, ylist_series=gasFdbk_vars_series, timeout_seconds=2*60)
    #func_run_with_multiprocessing(x=1, y=7, timeout_seconds=2*5)
