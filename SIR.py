#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 21:06:51 2021

@author: sambollman
"""

#https://community.plotly.com/t/dash-range-slider-which-able-to-interact-with-input-field-that-display-range-slider-value/49476
#https://codepen.io/chriddyp/pen/bWLwgP.css

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


# server = app.server()


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


app.layout = html.Div([
    
    html.Div(children=[
    
    html.H1(children='COVID-19 Model', style={'textAlign': 'center', 'color': '#034efc'}),
    # html.Div(children = '''SIR Graph'''),
    
    dcc.Graph(id= 'SIR-graph', figure=plotly_fig)]),
    
    # dcc.RadioItems(
    #             id='yaxis-type',
    #             options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
    #             value='Linear',
    #             labelStyle={'display': 'inline-block'}),
    
    html.Div(style={'columnCount': 3}, children=[
    
    html.Div(id='slider-output-container0'),
    
    dcc.Input(id="pop-input", type="number", min=1000, max=10000000,  placeholder="Enter population"),
    dcc.Slider(
        id='pop-slider',
        min=1000,
        max=10000000,
        step=1000,
        value=100000,
  
        
    ),
    
    # html.Div(id='slider-output-container1'),
    "Days:",
    dcc.Input(id="range_input", type="number", min=10, max=1000, placeholder="Enter days", style={'marginRight':'10px'}),
    dcc.Slider(
        id='range_slider',
        min=10,
        max=1000,
        step=1,
        value=200,
    ),

    html.Div(id='slider-output-container2'),
    dcc.Input(id="rate-input", type="number", min=.0001, max=1, placeholder="Enter infection rate", style={'marginRight':'10px'}),
    dcc.Slider(
        id='rate-slider',
        min=.0001,
        max=1,
        step=.01,
        value=5e-1,
    ),  
    html.Div(id='slider-output-container22'),
    dcc.Slider(
        id='rate-confidence-slider',
        min=0,
        max=.2,
        step=.01,
        value=0,
        
    ),
    html.Div(id='slider-output-container3'),
    dcc.Input(id="recovery-input", type="number", min=.01, max=1, placeholder="Enter recovery rate", style={'marginRight':'10px'}),
    dcc.Slider(
        id='recovery-slider',
        min=.01,
        max=1,
        step=.01,
        value=.1,
    ),
    html.Div(id='slider-output-container33'),
    dcc.Slider(
        id='recovery-confidence-slider',
        min=0,
        max=.2,
        step=.01,
        value=0,
        
    ),
    html.Div(id='slider-output-container4'),
    dcc.Input(id="immunity-input", type="number", min=.01, max=1,  placeholder="Enter immunity", style={'marginRight':'10px'}),
    dcc.Slider(
        id='immunity-slider',
        min=.01,
        max=1,
        step=.01,
        value=.1,      
    ),
    html.Div(id='slider-output-container44'),
    dcc.Slider(
        id='immunity-confidence-slider',
        min=0,
        max=.2,
        step=.01,
        value=0,
    
    ),
    html.Div(id='slider-output-container5'),
    dcc.Input(id="vaccinated-input", type="number", min=0, max=100, size="50", placeholder="Enter % vaccinated"),
    dcc.Slider(
        id='vaccinated-slider',
        min=0,
        max=100,
        step=.1,
        value=0,
        
    ),
    
    html.Div(id='slider-output-container6'),
    dcc.Input(id="mask-input", type="number", min=0, max=100,  placeholder="Enter % wearing masks", style={'marginRight':'10px'}),
    dcc.Slider(
        id='mask-slider',
        min=0,
        max=100,
        step=.1,
        value=0,
        
    ),
    
    ])
    
    ])


plotly_fig.update_layout(title={
                            'text': "SIR Graph",
                            'x':0.05},

            width=1000, height=500, 
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
    Output('pop-slider', 'value'),
    Output('pop-input', 'value'),
    Input('pop-slider', 'value'),
    Input('pop-input', 'value'))
def update_output00(s_value, i_value):
    if i_value is None:
        pass
    else:
        s_value = i_value
    return s_value, i_value

@app.callback(
    Output('slider-output-container0', 'children'),
    Input('pop-slider', 'value'))
def update_output0(value):
    return 'Population: "{}"'.format(value)


        

@app.callback(
    Output('range_slider', 'value'),
    Output('range_input', 'value'),
    Input('range_slider', 'value'),
    Input('range_input', 'value'))
def update_output11(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    
    
    if trigger_id == "range_slider" :
        range_slider = s_value
    else:
        range_slider = i_value
         
    if trigger_id == "range_input" :
          range_input = i_value 
    else :
        range_input = s_value
    
    return range_slider, range_input






@app.callback(
    Output('rate-slider', 'value'),
    Output('rate-input', 'value'),
    Input('rate-slider', 'value'),
    Input('rate-input', 'value'))
def update_output22(s_value, i_value):
    if i_value is None:
        pass
    else:
        s_value = i_value
    return s_value, i_value
@app.callback(
    Output('slider-output-container2', 'children'),
    Input('rate-slider', 'value'))
def update_output2(value):
    return 'Infection Rate: "{}"'.format(value)
@app.callback(
    Output('slider-output-container22', 'children'),
    Input('rate-confidence-slider', 'value'))
def update_output000(value):
    return 'Uncertianty: "{}"'.format(value)

@app.callback(
    Output('slider-output-container3', 'children'),
    Input('recovery-slider', 'value'))
def update_output3(value):
    return 'Recovery rate: "{}"'.format(value)
@app.callback(
    Output('slider-output-container33', 'children'),
    Input('recovery-confidence-slider', 'value'))
def update_output333(value):
    return 'Uncertianty: "{}"'.format(value)

@app.callback(
    Output('slider-output-container4', 'children'),
    Input('immunity-slider', 'value'))
def update_output4(value):
    return 'Post infection immunity: "{}"'.format(value)
@app.callback(
    Output('slider-output-container44', 'children'),
    Input('immunity-confidence-slider', 'value'))
def update_output444(value):
    return 'Uncertianty: "{}"'.format(value)

@app.callback(
    Output('slider-output-container5', 'children'),
    Input('vaccinated-slider', 'value'))
def update_output5(value):
    return 'Percent vaccinated: "{}"'.format(value)

@app.callback(
    Output('slider-output-container6', 'children'),
    Input('mask-slider', 'value'))
def update_output6(value):
    return 'Percent of people wearing masks in public: "{}"'.format(value)


@app.callback(
    Output('SIR-graph', 'figure'),
    Input('pop-slider', 'value'),
    Input('range_input', 'value'),
    Input('rate-slider', 'value'),
    # Input('yaxis-type', 'value'),
    Input('recovery-slider', 'value'),
    Input('immunity-slider', 'value'),
    Input('vaccinated-slider', 'value'),
    Input('mask-slider', 'value')
    )

def update_graph(pop_value, range_value, rate_value, recovery_value, 
                 immunity_value, vaccinated_value, mask_value):
    
    
    mask_value = abs(((mask_value/100)*.8)-1)
    
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

            width=1350, height=500, 
            paper_bgcolor='#222222',plot_bgcolor ='#e3e3e3', 
            xaxis_title="Days", yaxis_title="Population", 
            font=dict(size=16, color="#ffffff"))
    
    
    

    return plotly_fig





if __name__ == '__main__':
    app.run_server(port=1813,debug=True, host='localhost')
    
    
    
