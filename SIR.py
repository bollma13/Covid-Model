#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 21:06:51 2021

@author: sambollman
"""

import numpy as np
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


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])


server = app.server


def sir_derivs(time, y, beta_0, omega, gamma, epsilon, N):
   
    s = y[0]
    i = y[1]
    r = y[2]
   
    beta = beta_0*(np.sin(omega*time) + 1)
   
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


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solution.t, solution.y[0], label = 'S')
ax.plot(solution.t, solution.y[1], label = 'I')
ax.plot(solution.t, solution.y[2], label = 'R')
plt.xlabel("time")
ax.grid(True)
plotly_fig = mpl_to_plotly(fig)


app.layout = html.Div(
    children=[html.H1(children='COVID-19 Model', style={'textAlign': 'center', 'color': '#2969ff'}),
    # html.Div(children = '''SIR Graph'''),
    
    dcc.Graph(id= 'SIR-graph', figure=plotly_fig),
    
    # dcc.RadioItems(
    #             id='yaxis-type',
    #             options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
    #             value='Linear',
    #             labelStyle={'display': 'inline-block'}),
    html.Div(id='slider-output-container0'),
    dcc.Slider(
        id='pop-slider',
        min=1000,
        max=10000000,
        step=1000,
        value=100000,
    ),
    
    html.Div(id='slider-output-container1'),
    dcc.Slider(
        id='range-slider',
        min=10,
        max=1000,
        step=1,
        value=200,
    ),
    
    html.Div(id='slider-output-container2'),
    dcc.Slider(
        id='rate-slider',
        min=.001,
        max=.1,
        step=.001,
        value=.05,
    ),
    
    html.Div(id='slider-output-container3'),
    dcc.Slider(
        id='recovery-slider',
        min=.01,
        max=1,
        step=.01,
        value=.1,
        
    ),
    
 
    ])


plotly_fig.update_layout(title={
                            'text': "SIR Graph",
                            'x':0.05},

            width=1250, height=500, 
            paper_bgcolor='#636363',plot_bgcolor ='#e3e3e3', 
            xaxis_title="Days", yaxis_title="Population", 
            font=dict(size=18, color="#ffffff"))

plotly_fig.update_layout(updatemenus=[dict(buttons=[
                      dict(label="Linear",  
                          method="relayout", 
                          args=[{"yaxis.type": "linear"}]),
                      dict(label="Log", 
                          method="relayout", 
                          args=[{"yaxis.type": "log"}])])])



@app.callback(
    Output('slider-output-container0', 'children'),
    Input('pop-slider', 'value'))

def update_output0(value):
    return 'Population: "{}"'.format(value)

@app.callback(
    Output('slider-output-container1', 'children'),
    Input('range-slider', 'value'))

def update_output1(value):
    return 'Number of days: "{}"'.format(value)
    
@app.callback(
    Output('slider-output-container2', 'children'),
    Input('rate-slider', 'value'))

def update_output2(value):
    return 'Infection Rate: "{}"'.format(value)

@app.callback(
    Output('slider-output-container3', 'children'),
    Input('recovery-slider', 'value'))

def update_output3(value):
    return 'Recovery rate: "{}"'.format(value)


@app.callback(
    Output('SIR-graph', 'figure'),
    Input('pop-slider', 'value'),
    Input('range-slider', 'value'),
    Input('rate-slider', 'value'),
    # Input('yaxis-type', 'value'),
    Input('recovery-slider', 'value'))

def update_graph(pop_value, range_value, rate_value, recovery_value):
    
    time_range = [0, range_value]
    initial_conditions = [pop_value, 100, 0]
    # beta_0 = 5e-1
    gamma = recovery_value
    epsilon = 0.1 * recovery_value
    N = pop_value
    

    fine_time = np.linspace(time_range[0], time_range[1], 600)
    
    solution = solve_ivp(sir_derivs, time_range, initial_conditions, t_eval=fine_time, args = (beta_0, omega, gamma, epsilon, N))
    
   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(solution.t, solution.y[0], label = 'S')
    ax.plot(solution.t, solution.y[1], label = 'I')
    ax.plot(solution.t, solution.y[2], label = 'R')
    ax.grid(True)
    plotly_fig = mpl_to_plotly(fig)
    
    
    # if yaxis_value == 'Linear':
    #     plotly_fig.update_yaxes(type='linear' )
    
    # if yaxis_value == 'Log':
    #     plotly_fig.update_yaxes(type='log', range=[0,7])
    
    plotly_fig.update_layout(updatemenus=[dict(buttons=[
                      dict(label="Linear",  
                          method="relayout", 
                          args=[{"yaxis.type": "linear"}]),
                      dict(label="Log", 
                          method="relayout", 
                          args=[{"yaxis.type": "log"}])])])
    
    plotly_fig.update_layout(title={
                            'text': "SIR Graph",
                            'x':0.05},

            width=1250, height=500, 
            paper_bgcolor='#222222',plot_bgcolor ='#e3e3e3', 
            xaxis_title="Days", yaxis_title="Population", 
            font=dict(size=16, color="#ffffff"))
    
    
    

    return plotly_fig





if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
