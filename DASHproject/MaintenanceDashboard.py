from DashPreprocessing import func_import_data, func_preprocess_data
from DigiPythonTools import DataUtility
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback, ctx
#from dash.dependencies import Input, Output
#import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly
#import plotly.io as pio #plotly 5.5.0
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc
import time, datetime
import os
from scipy import stats as st
from datetime import date, timedelta
import pickle


####################Import Data#########################################
####CHECK STORED MODEL#####
# # Deserialization
# with open("dfXY.pickle", "rb") as infile:
#     dfXY = pickle.load(infile)
# #print("Reconstructed object", dfXY)
#
# dfDown = dfXY.iloc[[int(x) for x in list(np.round(np.linspace(0, dfXY.shape[0] - 1, 2000), 0))],:]  # Downsample evenly

try:
    global dfXY
    Init = True
    if Init:
        Init = False
        #os.chdir(r"C:\Users\js5296\PycharmProjects\ECP2_AP1")
        #dfXY=func_import_data((date.today()-timedelta(days=5)).strftime("%Y-%m-%d"),5)
        #dfXY = func_import_data('mocoroco', 5)
        #dfDown=dfXY.copy()
        #os.chdir(r"C:\Users\js5296\PycharmProjects\ECP2_AP1\Dash")
        # Deserialize (load) the DataFrame from the file
        original_dir = os.getcwd()
        new_dir = r".\DASHproject"  # Replace with the actual path .\DASHproject
        os.chdir(new_dir)
        with open('dfXY.pickle', 'rb') as f:
            df_loaded = pickle.load(f)
        os.chdir(original_dir)
        dfXY = df_loaded
        dfXY=dfXY.iloc[[int(x) for x in list(np.round(np.linspace(0,dfXY.shape[0]-1,2000),0))],:]#Downsample evenly
        dfXY.columns = dfXY.columns[:-1].tolist()+["SteelFamLabel"]
        #dfXY = func_preprocess_data(dfXY.copy())

        #CREATE VARIABLES
        numerical_var_series=dfXY[dfXY.columns[dfXY.dtypes == np.float64]].columns#.to_list()#[0:5]
        #Create the different groups of columns to plot
        proceso_vars_series=dfXY[numerical_var_series[(~numerical_var_series.str.contains("AP1_Furnace",regex=False)) & (~numerical_var_series.str.contains("pyro2",regex=False))]].columns#.head(1)
        zoneTemp_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Temp.*",regex=True)]].columns#.head(1)
        zoneAirGas_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*Zone.*Air.*|.*Zone.*Gas.*",regex=True)]].columns#.head(1)
        recuperator_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains("AP1_Furnace.*Dilution|Recuperator|RWP|Combustion.*",regex=True)]].columns#.head(1)
        pyro_vars_series=dfXY[numerical_var_series[numerical_var_series.str.contains(".*pyro.*|.*Pyro.*",regex=True)]].columns#.head(1)
        rest_vars_series=dfXY[[var for var in numerical_var_series.to_list() if var not in proceso_vars_series.to_list() and var not in zoneTemp_vars_series.to_list() and var not in zoneAirGas_vars_series.to_list() and var not in recuperator_vars_series.to_list() and var not in pyro_vars_series.to_list()]].columns#.head(1)

        gasFdbk_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*GasFlowF.*db.*k.*",regex=True)]
        gasFdbk_vars_series.name="gasFdbk_vars_series"
        airFdbk_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*AirFlowF.*db.*k.*", regex=True)]
        airFdbk_vars_series.name= "airFdbk_vars_series"
        airSP_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*AirFlow[Ss]{1}.*[Pp]{1}.*", regex=True)]
        airSP_vars_series.name= "airSP_vars_series"
        gasCV_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*GasFlowC.*V.*",regex=True)]
        gasCV_vars_series.name="gasCV_vars_series"
        airCV_vars_series=zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*AirFlowC.*V.*", regex=True)]
        airCV_vars_series.name="airCV_vars_series"

        AirGasRatioFdbk_vars_series = zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Ratio.*PV.*|.*Ratio.*Actual.*", regex=True)]
        AirGasRatioFdbk_vars_series.name = "AirGasRatioFdbk_vars_series"
        AirGasRatioSP_vars_series = zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Ratio.*[Ss].*[Pp].*", regex=True)]
        AirGasRatioSP_vars_series.name = "AirGasRatioSP_vars_series"

        tempFdbk_vars_series = zoneTemp_vars_series[zoneTemp_vars_series.str.contains(".*TopTemp.*|.*Fdbk*", regex=True)]
        tempFdbk_vars_series.name = "tempFdbk_vars_series"
        tempSP_vars_series = zoneTemp_vars_series[~zoneTemp_vars_series.str.contains(".*TopTemp.*|.*Fdbk.*|.*CV.*|.*Operator.*", regex=True)]
        tempSP_vars_series.name = "tempSP_vars_series"

        #Zone2 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
        dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]=dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]*47565/59655
except:
    alert = True
    alarm = "A backend error occured during the preprocessing of the data. \nPlease check file DashPreprocessing.py for troubleshooting and let the ring of power guide you. \nPossible error sources: DB problems or python libraries deprecated."
    #print("Ini alert True")
else:
    alert = False
    alarm = "An error occured. \nPlease try again later."
    #print("Ini alert False")


image_path = 'assets/background_updated.png'

############################Ini app###############################
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.title = 'Interactive Model Dashboard'
######################Define Dash Layout#################################
app.layout = html.Div(
    [
        dbc.Navbar(
                html.A(dbc.Row([
                    dbc.Col(html.Img(
                        src="https://nasnet.nasnet.biz/images/logo-nast.png",
                        height="45px")),
                    dbc.Col(dbc.NavbarBrand("AP1_Furnace Maintenance DashBoard", className="ml-2")),
                ], align="center"),
                    style={"marginLeft": "15px"}
                ),
                sticky='top',
        ),
        html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown',
                            options=[
                                {'label': 'GasFeedback vs GasCV', 'value': 'GasFeedback vs GasCV'},
                                {'label': 'AirFeedback vs AirCV', 'value': 'AirFeedback vs AirCV'},
                                {'label': 'AirSP vs AirCV', 'value': 'AirSP vs AirCV'},
                                {'label': 'AirSP vs AirFeedback', 'value': 'AirSP vs AirFeedback'},
                                {'label': 'AirGasRatioSP vs AirGasRatioFeedback', 'value': 'AirGasRatioSP vs AirGasRatioFeedback'},
                                {'label': 'TempSP vs TempFeedback', 'value': 'TempSP vs TempFeedback'}
                            ],
                            value='GasFeedback vs GasCV',
                            clearable=False
                        ), width=3
                        #style={'marginLeft': '20px', 'width': '20%', 'backgroundColor': 'orange'}, align='center'
                        # 'marginLeft': '20px',
                    ),
                    # dbc.Col(width="auto"),#width=10
                    dbc.Col(
                        dbc.Button("refresh", color="success", className="me-1", id="refresh-button", n_clicks=0,
                                   size="sm"),
                        #style={'marginLeft': 'auto', 'marginRight': '20px', 'width': '5%', 'backgroundColor': 'green'},
                        align='center',  # 'marginLeft': 'auto', 'marginRight': '20px',
                        width=3
                    ),
                    dbc.Col(
                            html.Div("Introduce a start date: "),
                    #style={'marginLeft': 'auto', 'marginRight': '20px', 'width': '5%', 'backgroundColor': 'green'},
                    align='center'  # 'marginLeft': 'auto', 'marginRight': '20px',
                    ),
                    dbc.Col(
                        dcc.DatePickerSingle(
                            id='start-date-picker',
                            min_date_allowed=date(2024, 7, 18),  # Date started populating SQL tbl
                            max_date_allowed=str(date.today()-timedelta(days=5)),
                            initial_visible_month=str(date.today()-timedelta(days=5)),
                            date=str(date.today()-timedelta(days=5)),
                            display_format='YYYY-MM-DD'
                        ),
                        align='center'
                    ),
                    dbc.Col(
                        html.Div("End date: "),
                        align='center'
                    ),
                    dbc.Col(
                        dcc.DatePickerSingle(
                            id='end-date',
                            date=str(date.today()),
                            display_format='YYYY-MM-DD',
                            disabled=True
                        ),
                        align='center'
                    ),
                ], #style={'backgroundColor': 'pink'}, align="center"  # , justify="between"
            ),
            dbc.Row([
                dbc.Alert(
                    children="An error occured. \nPlease try again later.",
                    id="alert-auto",
                    is_open=alert,
                    color='danger'
                ),
                dcc.Loading([
                    #html.Div("Loading data..."),
                    dcc.Graph(id='scatter-plot'),
                ],
                    type="graph", #graph, cube, circle, dot, default
                    target_components={"scatter-plot": "figure"},
                    overlay_style={"visibility":"visible", "opacity": .8, "backgroundColor": "white", "filter": "blur(2px)"}, #"filter": "blur(2px)"
                    custom_spinner=html.H2([dbc.Spinner(color="primary", spinner_style={"width": "2rem", "height": "2rem", "fontsize":"50px"})," Loading data..."]), #color="primary","success","info"
                ),
            ]
            )
        ])
    ])



#Callback for updating end date
@callback(
    Output('end-date', 'date'),
    [Input('start-date-picker', 'date')],
    prevent_initial_call=True,
)
def update_end_date(input_date_string):

    return str(datetime.datetime.strptime(input_date_string, '%Y-%m-%d').date()+timedelta(days=5))




#Callback for rendering graph
@callback(
    [Output('scatter-plot', 'figure'),
     Output('alert-auto', 'is_open'),
     Output('alert-auto', 'children')],
    [Input('dropdown', 'value'),
     Input('refresh-button', 'n_clicks'),
     Input('start-date-picker', 'date')],
    prevent_initial_call=True,
)

def update_figure(selected_option,refresh_button,start_date_string):
    try:
        # Refresh Data
        if ("refresh-button" == ctx.triggered_id) | ("start-date-picker" == ctx.triggered_id):
            global dfXY
            #dfXY = func_import_data(start_date_string,5)
            # Deserialize (load) the DataFrame from the file
            original_dir = os.getcwd()
            new_dir = r".\DASHproject"  # Replace with the actual path 
            os.chdir(new_dir)
            with open('dfXY.pickle', 'rb') as f:
                df_loaded = pickle.load(f)
            os.chdir(original_dir)
            dfXY = df_loaded
            dfXY=dfXY.iloc[[int(x) for x in list(np.round(np.linspace(0,dfXY.shape[0]-1,2000),0))],:]#Downsample evenly
            dfXY.columns = dfXY.columns[:-1].tolist()+["SteelFamLabel"]
            #dfXY = func_preprocess_data(dfXY.copy())
            #Zone2 air SP are scaled with outdated scale, rescale:outdated scale: 47565 SCFH->updated scale: 59655 SCFH
        dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]=dfXY.loc[:,zoneAirGas_vars_series[zoneAirGas_vars_series.str.contains(".*Zone2.*AirFlowSet[Pp]oint.*",regex=True)]]*47565/59655
            #print("Refresh-calendar changed")
            #dfXY = func_import_data('mocoroco', 5)
            #dfDown = dfXY.copy()
    except:
        #print("Callback backend alarm True")
        alarm = "A backend error occured during the preprocessing of the data. \nPlease check file DashPreprocessing.py for troubleshooting and let the ring of power guide you. \nPossible error sources: DB problems or python libraries deprecated."
        return {}, True, alarm #dash.no_update
    else:
        try:
            #print("Callback backend alarm False")
            #print(1/0)
            #Define xlist_series and ylist_series
            if selected_option == 'GasFeedback vs GasCV':
                xlist_series=gasCV_vars_series
                ylist_series=gasFdbk_vars_series
            elif selected_option == 'AirFeedback vs AirCV':
                xlist_series = airCV_vars_series
                ylist_series = airFdbk_vars_series
            elif selected_option == 'AirSP vs AirCV':
                xlist_series = airCV_vars_series
                ylist_series = airSP_vars_series
            elif selected_option == 'AirSP vs AirFeedback':
                xlist_series = airFdbk_vars_series
                ylist_series = airSP_vars_series
            elif selected_option == 'AirGasRatioSP vs AirGasRatioFeedback':
                xlist_series = AirGasRatioFdbk_vars_series
                ylist_series = AirGasRatioSP_vars_series
            elif selected_option == 'TempSP vs TempFeedback':
                xlist_series = tempFdbk_vars_series
                ylist_series = tempSP_vars_series

            #x, y = selected_option.split(' vs ')
            #fig = px.scatter(df, x=x, y=y, title=selected_option)

            ncol = 9
            nrow = 2
            var = 0
            subtitles_list = []
            for i in range(2):  # Bottom/Top
                for j in range(9):  # 9zones
                    if i == 0:
                        subtitles_list.append(f"Zone{j} Top")
                    else:
                        subtitles_list.append(f"Zone{j} Bottom")

            if {"tempSP_vars_series", "tempFdbk_vars_series"}.isdisjoint([xlist_series.name, ylist_series.name]):
                fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=subtitles_list,
                                    vertical_spacing=0.1)  # vertical_spacing(0-1)
            else: #plot Temp vars
                fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=subtitles_list[:9],
                                    vertical_spacing=0.1)  # vertical_spacing(0-1)

            for r in range(1, nrow + 1):
                # How to plot vars
                if {"tempSP_vars_series", "tempFdbk_vars_series"}.isdisjoint([xlist_series.name, ylist_series.name]):
                    if r == 1:
                        var = 1
                    else:
                        var = 0
                else:
                    var = 0
                    if r >= nrow:
                        break
                for c in range(1, ncol + 1):
                    if {"AirGasRatioFdbk_vars_series", "AirGasRatioSP_vars_series"}.isdisjoint(
                            [xlist_series.name, ylist_series.name]):
                        # Create x,y,yhat
                        x = np.asarray(dfXY[xlist_series[var]]).reshape(-1, 1)
                        y = np.asarray(dfXY[ylist_series[var]])
                        y_pred = LinearRegression().fit(x, y).predict(x)
                        x = x[:, 0]
                    # plot Scatters
                    if (r == 1) & (c == 1):
                        fig.add_trace(go.Scatter(x=dfXY[dfXY.SteelFamLabel == "HighGoal"][xlist_series[var]],
                                                 y=dfXY[dfXY.SteelFamLabel == "HighGoal"][ylist_series[var]],
                                                 mode='markers',
                                                 # zorder=3,
                                                 name="HighGoal",
                                                 legendgroup="group1",
                                                 marker=dict(color="blue", line_width=0.6)
                                                 # ,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                                 ), row=r, col=c)
                        fig.add_trace(go.Scatter(x=dfXY[dfXY.SteelFamLabel == "LowGoal"][xlist_series[var]],
                                                 y=dfXY[dfXY.SteelFamLabel == "LowGoal"][ylist_series[var]],
                                                 mode='markers',
                                                 # zorder=3,
                                                 name="LowGoal",
                                                 legendgroup="group2",
                                                 marker=dict(color="green", line_width=0.6)
                                                 # ,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                                 ), row=r, col=c)
                    else:
                        fig.add_trace(go.Scatter(x=dfXY[dfXY.SteelFamLabel == "HighGoal"][xlist_series[var]],
                                                 y=dfXY[dfXY.SteelFamLabel == "HighGoal"][ylist_series[var]],
                                                 mode='markers',
                                                 # zorder=3,
                                                 name="HighGoal",
                                                 legendgroup="group1",
                                                 marker=dict(color="blue", line_width=0.6)
                                                 # ,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                                 , showlegend=False), row=r, col=c)
                        fig.add_trace(go.Scatter(x=dfXY[dfXY.SteelFamLabel == "LowGoal"][xlist_series[var]],
                                                 y=dfXY[dfXY.SteelFamLabel == "LowGoal"][ylist_series[var]],
                                                 mode='markers',
                                                 # zorder=3,
                                                 name="LowGoal",
                                                 legendgroup="group2",
                                                 marker=dict(color="green", line_width=0.6)
                                                 # ,marker=dict(color=dfZoneXScatter.SteelFamCluster.map({"HighGoal":"blue","LowGoal":"green"}),line_width=0.6)
                                                 , showlegend=False), row=r, col=c)

                    if {"AirGasRatioFdbk_vars_series", "AirGasRatioSP_vars_series"}.isdisjoint(
                            [xlist_series.name, ylist_series.name]):
                        # plot regression line
                        fig.add_trace(go.Scatter(x=x,
                                                 y=y_pred,
                                                 mode='lines',
                                                 # zorder=4,
                                                 # name='2',
                                                 line=dict(width=1.5, color="orange"),
                                                 showlegend=False), row=r, col=c)
                        # plot regression data
                        fig.add_annotation(x=dfXY[xlist_series[var]].min(), y=dfXY[ylist_series[var]].max() + 10,
                                           # text="r\u00b2={:.2f}, p={:.2E}".format(st.pearsonr(x,y)[0]**2,st.pearsonr(x,y)[1]),
                                           text="r\u00b2={:.2f}".format(st.pearsonr(x, y)[0] ** 2),
                                           # hovertext="pepe",
                                           align='center',
                                           showarrow=False,
                                           xanchor='left',
                                           yanchor='bottom',
                                           # clicktoshow="onout",
                                           # captureevents=True,
                                           # xref='x2',
                                           # yref='y2',
                                           row=r,
                                           col=c,
                                           bordercolor='black',
                                           borderpad=4,
                                           borderwidth=2,
                                           bgcolor='white',
                                           font=dict(color='black'))
                    # Add xaxis labels
                    # fig.update_xaxes(title_text=xlist_series[var], row=r, col=c)#range=[40, 80],showgrid=False,type="log",
                    # Add yaxis labels
                    # fig.update_yaxes(title_text=ylist_series[var], row=r, col=c)

                    # update var to plot
                    if {"tempSP_vars_series", "tempFdbk_vars_series"}.isdisjoint([xlist_series.name, ylist_series.name]):
                        var += 2
                    else:
                        var += 1

            # Add title and size and show img
            fig.update_layout(height=900, width=1850,
                              title_text=f'x={xlist_series.name} vs y={ylist_series.name} across all zones from {dfXY.index[0]} to {dfXY.index[-1]}', showlegend=True,
                              title_x=0.5)  # height=500*9, width=1000, #, legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01)
        except:
            #print("Callback frontend alarm True")
            alarm ="A frontend error occured during the rendering of the app. \nRohan will answer, meanwhile check file MaintenanceDasboard.py for troubleshooting. \nPossible error sources: DB problems or python libraries deprecated."
            return {}, True, alarm
        else:
            #print("Callback frontend alarm False")
            alarm = "An error occured. PLease try again later."
            return fig, False, alarm

server = app.server

#################RUN APP############################
if __name__ == '__main__':
    app.run_server(debug=False,
                   port=8050,
                   host='0.0.0.0')