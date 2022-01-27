#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 21:06:51 2021

@author: sambollman
#https://community.plotly.com/t/dash-range-slider-which-able-to-interact-with-input-field-that-display-range-slider-value/49476
#https://codepen.io/chriddyp/pen/bWLwgP.css
#https://chrisalbon.com/
"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
from scipy.integrate import solve_ivp
import dash 
import dash_html_components as html
import dash_core_components as dcc
# import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq

# from flask import request, Response


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])
server = app.server


def sir_derivs(time, y, beta_0, omega, gamma, epsilon, N):
   
    s = y[0]
    i = y[1]
    r = y[2]
   
    beta = beta_0
    # *(np.sin(omega*time) + 1)
   
    dsdt = -beta*s*i/N + epsilon*r
    didt = beta*s*i/N - gamma*i
    drdt = gamma*i - epsilon*r
   
    return [dsdt, didt, drdt]


time_range = [0, 200]
initial_conditions = [100000, 100, 0]
beta_0 = 5e-1
omega = 2*np.pi/365
gamma = 0.1
epsilon = 0.01
N = 100000


fine_time = np.linspace(time_range[0], time_range[1], 600)
solution = solve_ivp(sir_derivs, time_range, initial_conditions, t_eval=fine_time, args = (beta_0, omega, gamma, epsilon, N))

plt.plot(solution.t, solution.y[0], label = 'S')
plt.plot(solution.t, solution.y[1], label = 'I')
plt.plot(solution.t, solution.y[2], label = 'R')
plt.xlabel("time")


fig = plt.Figure()
ax = fig.add_subplot(111)
ax.plot(solution.t, solution.y[0], label = 'S', linewidth=3, color='#005d8f')
ax.plot(solution.t, solution.y[1], label = 'I', linewidth=3, color='#8f71eb')
ax.plot(solution.t, solution.y[2], label = 'R', linewidth=3, color='#de4e4e')  
plt.xlabel("time")
ax.grid(True)
ax.legend(["Susceptible", "Infected", "Recovered"],  loc='best', prop={"size":15}, labelcolor='linecolor')

plotly_fig = mpl_to_plotly(fig)




app.layout = html.Div([
        html.Div(children = 'COVID-19 Model', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 40, 'padding': "5px"}),
        dcc.Tabs([

    dcc.Tab(label='Controls', 
        children=[
         
    html.Div(style={'textAlign': 'center','display': 'flex', 'flex-direction': 'row'}, children=[
        
        html.Div(style={'padding': 10, 'flex': 1},children=[
        html.Div(children = 'Welcome!', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
        html.Label('This web app was developed to model COVID-19 using an SIR model. Read the instructions to get started or find more information on the \'More Info\' tab.', 
                   style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px"}),
        html.Div(children = 'Instructions', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
        html.Label('Use the text boxes and sliders to edit values on the right. Click the \'Plots\' tab at the top to see the result. You can find more information on the \'More Info\' tab.', 
                   style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px"}),
        html.Div(children = 'r0', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
        html.Label('r0 or r-naught represents the average amount of people that a single person will infect with a disease. In this model r0 is shown by dividing the infection rate by the recovery rate.',
                   style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px"}),
        html.Div(children =  'r0 values for common diseases(NPR):' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px", 'marginTop' : '5px'}),
        html.Div(children =  'influenza: 1-2' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'ebola: 2' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'HIV: 4' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'SARS: 4' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'mumps: 10' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'measles: 18' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'r0 values for COVID-19(BBC): ' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px", 'marginTop' : '10px'}),
        html.Div(children =  'original virus: 2.4-2.6' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'delta varient: 5-8' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
        html.Div(children =  'omnicron varient: unknown (likely >8)' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
    ]),
         
    html.Div(style={'textAlign': 'center','padding': 15, 'flex': 1},children=[
    html.Div(children = 'compare', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    html.Div([
                daq.ToggleSwitch(label='', color='gray',
                                 labelPosition='bottom',
                    id='compare_toggle',
                    value=False
                    ),
                html.Div(id='compare_toggle_output', style={'textAlign': 'center', 'color': '#bbbbbb'})
                ]),        
    # Population input, slider
    html.Div(children = 'population', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="pop_input", type="number", min=1000, max=10000000,  value=100000, placeholder="Population", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='pop_slider',
        min=1000,
        max=10000000,
        step=1000,
        value=100000,
        marks={
            1000: {'label': '1000', 'style': {'color': '#999999', 'fontSize': 15}},
            10000000: {'label': '10000000', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    
    
    # Range(days) input, slider
    html.Div(children = 'days', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="range_input", type="number", min=10, max=1000, value=200, placeholder="Days", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='range_slider',
        min=10,
        max=1000,
        step=1,
        value=200,
        marks={
            10: {'label': '10', 'style': {'color': '#999999', 'fontSize': 15}},
            1000: {'label': '1000', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
 

    # Vaccinated input, slider 
    html.Div(children = 'percent vaccinated', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="vaccinated_input", type="number", min=0, max=100, value=0, size="50", placeholder="% vaccinated", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='vaccinated_slider',
        min=0,
        max=100,
        step=.1,
        value=0,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            100: {'label': '100', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    
    # Mask input, slider 
    html.Div(children = 'percent wearing masks in public', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="mask_input", type="number", min=0, max=100,  value=0, placeholder="% wearing masks", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='mask_slider',
        min=0,
        max=100,
        step=.1,
        value=0,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            100: {'label': '100', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    
    html.Div(style={'padding': 15, 'flex': 1},children=[
        html.Label(children = '', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px"}),
        html.Div("r0", id="r0_output", style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px"})
            ])
    ]),
    
    html.Div(style={'padding': 15, 'flex': 1}, children=[

 # Infection Rate input, slider
    html.Div(children = 'infection rate (β)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="rate_input", type="number", min=.0001, max=1, value=5e-1, placeholder="Infection rate", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='rate_slider',
        min=0.01,
        max=1,
        step=.01,
        value=5e-1,
        marks={
            0.01: {'label': '0.01', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1.0', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),          # Infection Rate Confidence slider
    html.Div(id='slider-output-container22', style={'fontSize': 20}),
    dcc.Slider(
        id='rate-confidence-slider',
        min=0,
        max=1,
        step=.001,
        value=0,
        tooltip={"placement": "bottom"},
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ), 
        
    
    # Recovery input, slider
    html.Div(children = 'recovery rate (γ)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="recovery_input", type="number", min=.01, max=1, value=.1, placeholder="Recovery rate", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='recovery_slider',
        min=0.01,
        max=1,
        step=.01,
        value=.1,
        marks={
            .01: {'label': '0.01', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1.0', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),          #  Recovery confidence slider
    html.Div(id='slider-output-container33', style={'fontSize': 20}),
    dcc.Slider(
        id='recovery-confidence-slider',
        min=0,
        max=1,
        step=.001,
        value=0,
        tooltip={"placement": "bottom"},
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    
    
    # Immunity input, slider
    html.Div(children = 'post infection immunity', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
    dcc.Input(debounce=True, id="immunity_input", type="number", min=.01, max=1,  value=.1, placeholder="Immunity", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='immunity_slider',
        min=0.01,
        max=1,
        step=.01,
        value=.1,
        marks={
            .01: {'label': '0.01', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1.0', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),      # Immunity confidence slider
    html.Div(id='slider-output-container44', style={'fontSize': 20}),
    dcc.Slider(
        id='immunity-confidence-slider',
        min=0,
        max=1,
        step=.001,
        value=0,
        tooltip={"placement": "bottom"},
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    
    ])
    
    ])
    
    ]),
    
    dcc.Tab(label='Plots', children=[
        html.Div(children=[
            
    html.H1(
    # children='COVID-19 Model', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 30}
    ),
    # html.Div(children = 'COVID-19 Model', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 40, 'padding': "1px"}),
    html.Div(
    style={'display': 'inline-block'}),
    
    html.Div(style={'padding': 15, 'display': 'flex', 'flex-direction': 'row'},children=[
        
        html.Div(style={'padding': 10, 'flex': 1},children=[
            # html.Label(children = 'Graph 1', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 30, 'padding': "0px", 'marginTop' : '0px'}),
            dcc.Graph(id='SIR-graph', figure=plotly_fig, responsive=True, style={'display': 'inline-block', 'width': "80vw", 'height': '60vh'}),
            
            # html.Label(children = 'Graph 2', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 30, 'padding': "0px", 'marginTop' : '0px'}),
            dcc.Graph(id='SIR-graph2', figure=plotly_fig, responsive=True, style={'display': 'inline-block', 'width': "80vw", 'height': '60vh'}),
            
            
            ]),
        
        html.Div(style={'padding': 10, 'flex': 1},children=[
        html.Label(children = 'Note: hover over graph to see Plotly graph controls on top right', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 17, 'padding': "0px"}),
        html.Label(children = 'Graph 1 controls:', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px", 'marginTop' : '30px'}),
        html.Div("rate_out", id="rate_output", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("recovery_out", id="recovery_output", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("immunity_out", id="immunity_output", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("vax_out", id="vax_output", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("masks_out", id="masks_output", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("r0", id="r0_output2", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),

        html.Label(children = 'Graph 2 controls:', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px", 'marginTop' : '270px'}),
        html.Div("rate_out", id="rate_output2", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("recovery_out", id="recovery_output2", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("immunity_out", id="immunity_output2", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("vax_out", id="vax_output2", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("masks_out", id="masks_output2", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        html.Div("r0", id="r0_output3", style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 20, 'padding': "2px"}),
        
        
            
            # html.A(
            # "download CSV",
            # id="download_csv",
            # href="#",
            # className="btn btn-outline-secondary btn-sm",
            # style={'textAlign': 'center', 'color': '#ffffff', 'padding': '8px', 'fontSize': 20, 'borderColor':"#ffffff", 'marginTop' : '30px'},
            # )
            
            ])
    
    ]),
    
    ])
        
    ]),
    
    
    
    
    # More info Tab
    dcc.Tab(label='More Info', children=[
        html.Div(style={'textAlign': 'center','display': 'flex', 'flex-direction': 'row'}, children=[
        
            html.Div(style={'padding': 10, 'flex': 1}, children=[
                

                
                

                
                html.Div(children = 'Controls', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
                html.Label('r0 is given by infection rate(β) over recovery rate(γ).', 
                           style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                html.Div( style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px"}),
                html.Label('Recovery rate is given by 1 over the mean recovery time in days.', 
                           style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                ]),
                
            html.Div(style={'padding': 10, 'flex': 1}, children=[
                html.Div(children = 'Info', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
                html.Label('Information', 
                           style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "15px"}),
                html.Div(children = 'Created By', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
                html.Label('Sam Bollman, Michigan State University', 
                            style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "15px"}),
                ])
      
    ])  
       
    ])            
         
    
    ],
    style={'fontSize': 22, 'padding': '2px', 'width': '50%', 'margin': 'auto'},
    colors={"border": "white","primary": "#white","background": "#1a232d"},
    )],
    style={'textAlign': 'center'})

# Update graph initially: Add log/lin buttons, size, color, legend, axis
plotly_fig.update_layout(
                    updatemenus=[dict(buttons=[
                    dict(
                        label="linear",  
                        method="relayout", 
                        args=[{"yaxis.type": "linear"}]),
                    dict(
                        label="log", 
                        method="relayout", 
                        args=[{"yaxis.type": "log"}])],
                        showactive=False,
                        )],
                    xaxis = dict(
                        tickfont=dict(size=15)),
                    yaxis = dict(
                        tickfont=dict(size=15)),
                    
                    autosize=True,
                    paper_bgcolor='#222222',
                    plot_bgcolor ='#e3e3e3', xaxis_title="days", 
                    yaxis_title="population", 
                    font=dict(
                        size=22, color="#ffffff")    
        )




# Population input, slider sync
@app.callback(
    Output('pop_slider', 'value'),
    Output('pop_input', 'value'),
    Input('pop_slider', 'value'),
    Input('pop_input', 'value'))
def update_population_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        
    if trigger_id == "pop_slider" :
        pop_slider = s_value
    else:
        pop_slider = i_value
    
    if trigger_id == "pop_input":
        pop_input = i_value 
    else:
        pop_input = s_value
            
        
    if i_value != None: 
        return pop_slider, pop_input
    else:
        return s_value, s_value
        
# Range(days) input, slider sync
@app.callback(
    Output('range_slider', 'value'),
    Output('range_input', 'value'),
    Input('range_slider', 'value'),
    Input('range_input', 'value'))
def update_range_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "range_slider" :
        range_slider = s_value
    else:
        range_slider = i_value
         
    if trigger_id == "range_input" :
          range_input = i_value 
    else:
        range_input = s_value
    
    if i_value != None: 
        return range_slider, range_input
    else:
        return s_value, s_value

# Infection rate input, slider sync
@app.callback(
    Output('rate_slider', 'value'),
    Output('rate_input', 'value'),
    Input('rate_slider', 'value'),
    Input('rate_input', 'value'))
def update_rate_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "rate_slider" :
        rate_slider = s_value
    else:
        rate_slider = i_value
         
    if trigger_id == "rate_input" :
          rate_input = i_value 
    else:
        rate_input = s_value
    
    if i_value != None: 
        return rate_slider, rate_input
    else:
        return s_value, s_value
@app.callback(
    Output('slider-output-container22', 'children'),
    Input('rate-confidence-slider', 'value'))
def update_confidence_output1(value):
    return 'standard deviation: {}'.format(value), ''

# Recovery rate input, slider sync
@app.callback(
    Output('recovery_slider', 'value'),
    Output('recovery_input', 'value'),
    Input('recovery_slider', 'value'),
    Input('recovery_input', 'value'))
def update_recovery_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "recovery_slider" :
        recovery_slider = s_value
    else:
        recovery_slider = i_value
         
    if trigger_id == "recovery_input" :
          recovery_input = i_value 
    else:
        recovery_input = s_value
    
    if i_value != None: 
        return recovery_slider, recovery_input
    else:
        return s_value, s_value
@app.callback(
    Output('slider-output-container33', 'children'),
    Input('recovery-confidence-slider', 'value'))
def update_confidence_output2(value):
    return 'standard deviation: {}'.format(value), ''

# immunity  input, slider sync
@app.callback(
    Output('immunity_slider', 'value'),
    Output('immunity_input', 'value'),
    Input('immunity_slider', 'value'),
    Input('immunity_input', 'value'))
def update_immunity_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "immunity_slider" :
        immunity_slider = s_value
    else:
        immunity_slider = i_value
         
    if trigger_id == "immunity_input" :
          immunity_input = i_value 
    else:
        immunity_input = s_value
    
    if i_value != None: 
        return immunity_slider, immunity_input
    else:
        return s_value, s_value
@app.callback(  # immunity  confidence level output
    Output('slider-output-container44', 'children'),
    Input('immunity-confidence-slider', 'value'))
def update_confidence_output3(value):
    return 'standard deviation: {}'.format(value), ''

# Vaccinated  input, slider sync
@app.callback(
    Output('vaccinated_slider', 'value'),
    Output('vaccinated_input', 'value'),
    Input('vaccinated_slider', 'value'),
    Input('vaccinated_input', 'value'))
def update_vaccinated_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "vaccinated_slider" :
        vaccinated_slider = s_value
    else:
        vaccinated_slider = i_value
         
    if trigger_id == "vaccinated_input" :
          vaccinated_input = i_value 
    else:
        vaccinated_input = s_value
    
    if i_value != None: 
        return vaccinated_slider, vaccinated_input
    else:
        return s_value, s_value

# Masked  input, slider sync
@app.callback(
    Output('mask_slider', 'value'),
    Output('mask_input', 'value'),
    Input('mask_slider', 'value'),
    Input('mask_input', 'value'))
def update_mask_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "mask_slider" :
        mask_slider = s_value
    else:
        mask_slider = i_value
    if trigger_id == "mask_input" :
          mask_input = i_value 
    else:
        mask_input = s_value
    
    if i_value != None: 
        return mask_slider, mask_input
    else:
        return s_value, s_value

@app.callback(
    Output('compare_toggle_output', 'children'),
    Input('compare_toggle', 'value')
)
def update_output(value):
    if value == False:
        value = "top graph"
    else:
        value = "bottom graph"
    return 'you are editing the {}.'.format(value)



@app.callback(  # print r0 (betta/gamma)
    Output('r0_output', 'children'),
    Output('r0_output2', 'children'),
    Output('rate_output', 'children'),
    Output('recovery_output', 'children'),
    Output('immunity_output', 'children'),
    Output('vax_output', 'children'),
    Output('masks_output', 'children'),
    
    Output('r0_output3', 'children'),
    Output('rate_output2', 'children'),
    Output('recovery_output2', 'children'),
    Output('immunity_output2', 'children'),
    Output('vax_output2', 'children'),
    Output('masks_output2', 'children'),
    
    Input('rate_slider', 'value'),
    Input('recovery_slider', 'value'),
    Input('immunity_slider', 'value'),
    Input('vaccinated_slider', 'value'),
    Input('mask_slider', 'value'),
    Input('compare_toggle', 'value')

    )
def print_values(rate_value, recovery_value, immunity_value, vax_value, mask_value, graph_toggle):
    value = round(rate_value/recovery_value, 2)

    if graph_toggle == False:
        return 'r0 (β/γ) ≈ ''{}'.format(value), 'r0 ≈ ''{}'.format(value), "infection rate: " "{}" .format(rate_value), "recovery rate: " "{}" .format(recovery_value), "immunity: " "{}" .format(immunity_value), "percent vaxinated : " "{}" .format(vax_value), "percent masked: " "{}" .format(mask_value), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return 'r0 (β/γ) ≈ ''{}'.format(value), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'r0 ≈ ''{}'.format(value), "infection rate: " "{}" .format(rate_value), "recovery rate: " "{}" .format(recovery_value), "immunity: " "{}" .format(immunity_value), "percent vaxinated : " "{}" .format(vax_value), "percent masked: " "{}" .format(mask_value)




# Update Graph callback, interaction
@app.callback(
    Output('SIR-graph', 'figure'),
    Output('SIR-graph2', 'figure'),
    Input('pop_slider', 'value'),
    Input('range_input', 'value'),
    Input('rate_slider', 'value'),
    Input('recovery_slider', 'value'),
    Input('immunity_slider', 'value'),
    Input('vaccinated_slider', 'value'),
    Input('mask_slider', 'value'),
    Input('rate-confidence-slider', 'value'),
    Input('recovery-confidence-slider', 'value'),
    Input('immunity-confidence-slider', 'value'),
    Input('compare_toggle', 'value') 

    )

def update_graph(pop_value, range_value, rate_value, recovery_value, 
                 immunity_value, vaccinated_value, mask_value, rate_confidence_value,
                 recovery_confidence_value, immunity_confidence_value, graph_toggle):
    
    mask_value = abs(((mask_value/100)*.75)-1)
    vaccinated_value = abs(((vaccinated_value/100)*.9)-1)
    
    time_range = [0, range_value]
    initial_conditions = [pop_value, 100, 0]
    beta_0 = rate_value * mask_value * vaccinated_value
    gamma = recovery_value
    epsilon = immunity_value * recovery_value 
    N = pop_value

    fine_time = np.linspace(time_range[0], time_range[1], 600)
    solution = solve_ivp(sir_derivs, time_range, initial_conditions, t_eval=fine_time, args = (beta_0, omega, gamma, epsilon, N))
   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(solution.t, solution.y[0], label = 'S', linewidth=3, color='#005d8f')
    ax.plot(solution.t, solution.y[1], label = 'I', linewidth=3, color='#8f71eb')
    ax.plot(solution.t, solution.y[2], label = 'R', linewidth=3, color='#de4e4e')
        
    beta_d = np.sqrt(6)*rate_confidence_value
    rand_beta = np.random.triangular((beta_0 - beta_d)*.9999999999999, beta_0, (beta_0 + beta_d),size=10)

    gamma_d = np.sqrt(6)*recovery_confidence_value
    rand_gamma = np.random.triangular((gamma - gamma_d)*.9999999999999, gamma, (gamma + gamma_d), size=10)

    epsilon_d = np.sqrt(6)*immunity_confidence_value
    rand_epsilon = np.random.triangular((epsilon - epsilon_d)*.9999999999999, epsilon, (epsilon + epsilon_d), size=10)
    print(rand_epsilon + epsilon)  

    
    
    for a, b, c in zip(rand_beta, rand_gamma, rand_epsilon):
        rand_solution = solve_ivp(sir_derivs, time_range, initial_conditions, t_eval=fine_time, args = (a, omega, b, c, N))
        ax.plot(solution.t, rand_solution.y[0], color='#80aec7', linewidth=3)
        ax.plot(solution.t, rand_solution.y[1], color='#bcaaf3', linewidth=3)
        ax.plot(solution.t, rand_solution.y[2], color='#eb9595', linewidth=3)
        

    ax.plot(solution.t, solution.y[0], label = 'S', linewidth=3, color='#005d8f')
    ax.plot(solution.t, solution.y[1], label = 'I', linewidth=3, color='#8f71eb')
    ax.plot(solution.t, solution.y[2], label = 'R', linewidth=3, color='#de4e4e')  
   
    ax.grid(True)
    ax.legend(["Susceptible", "Infected", "Recovered"],  loc='best', prop={"size":15}, labelcolor='linecolor')
    
    plotly_fig = mpl_to_plotly(fig)
    
    plotly_fig.update_layout(
                    updatemenus=[dict(buttons=[
                    dict(
                        label="linear",  
                        method="relayout", 
                        args=[{"yaxis.type": "linear"}]),
                    dict(
                        label="log", 
                        method="relayout", 
                        args=[{"yaxis.type": "log"}])],
                        showactive=False,
                        )],
                    xaxis = dict(
                        tickfont=dict(size=15)),
                    yaxis = dict(
                        tickfont=dict(size=15)),
                    
                    autosize=True,
                    paper_bgcolor='#222222',
                    plot_bgcolor ='#e3e3e3', xaxis_title="days", 
                    yaxis_title="population", 
                    font=dict(
                        size=22, color="#ffffff")  
        )    
    
    
    if graph_toggle == False:
        return plotly_fig, dash.no_update
    else:
        return  dash.no_update, plotly_fig
    

    
     


# Update max for standard deviation sliders
@app.callback(  # Update max for the infection rate standard deviation
    Output('rate-confidence-slider', 'max'),
    Output('rate-confidence-slider', 'marks'),
    Output('rate-confidence-slider', 'value'),
    Input('rate_slider', 'value'),
    Input('vaccinated_slider', 'value'),
    Input('mask_slider', 'value'),

    )
    
def update_rate_confidence_slider(value1, value2, value3):
    value = (value1 * (abs(((value3/100)*.75)-1)) * (abs(((value2/100)*.9)-1)))/np.sqrt(6)
    marks = {0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            value: {'label': str(round(value,4)), 'style': {'color': '#999999', 'fontSize': 15}}
            }
    return value, marks, 0

@app.callback(  # Update max for the recovery rate standard deviation
    Output('recovery-confidence-slider', 'max'),
    Output('recovery-confidence-slider', 'marks'),
    Output('recovery-confidence-slider', 'value'),
    Input('recovery_slider', 'value')
    )
    
def update_recovery_confidence_slider(value):
    value = value/np.sqrt(6)
    marks = {0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            value: {'label': str(round(value,4)), 'style': {'color': '#999999', 'fontSize': 15}}
            }
    return value, marks, 0

@app.callback(  # Update max for the post infection immunity standard deviation
    Output('immunity-confidence-slider', 'max'),
    Output('immunity-confidence-slider', 'marks'),
    Output('immunity-confidence-slider', 'value'),
    Input('immunity_slider', 'value'),
    Input('recovery_slider', 'value'),

    )
     
def update_immunity_confidence_slider(value1, value2):
    value = (value1 * value2) /np.sqrt(6)
    marks = {0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            value: {'label': str(round(value,4)), 'style': {'color': '#999999', 'fontSize': 15}}
            }
    print(value) 
    return value, marks, 0
  
 



if __name__ == '__main__':
    app.run_server(debug = True)
    
    
    
    
