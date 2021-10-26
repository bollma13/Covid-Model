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
ax.legend(["Susceptible", "Infected", "Recovered"])

plotly_fig = mpl_to_plotly(fig)


app.layout = html.Div([
        html.Div(children=[
            
    html.H1(children='COVID-19 Model', style={'textAlign': 'center', 'color': '#034efc'}),
    # html.Div(children = '''SIR Graph'''),
    
    dcc.Graph(id= 'SIR-graph', figure=plotly_fig),
    
    html.Button("Download CSV", id="btn_csv"),
    dcc.Download(id="download-dataframe-csv"),
    
    ]),

    
    
    html.Div(style={'columnCount': 3}, children=[
        html.Div(children=[
            
    # Population input, slider
    "Population: ",
    dcc.Input(id="pop_input", type="number", min=1000, max=10000000,  value=100000, placeholder="Population"),
    dcc.Slider(
        id='pop_slider',
        min=1000,
        max=10000000,
        step=1000,
        value=100000,
        marks={
            1000: {'label': '1000', 'style': {'color': '#77b0b1'}},
            10000000: {'label': '10000000', 'style': {'color': '#f50'}}
            },
    ),
    
    # Range(days) input, slider
    "Days: ",
    dcc.Input(id="range_input", type="number", min=10, max=1000, value=200, placeholder="Days", style={'marginRight':'10px'}),
    dcc.Slider(
        id='range_slider',
        min=10,
        max=1000,
        step=1,
        value=200,
        marks={
            10: {'label': '10', 'style': {'color': '#77b0b1'}},
            1000: {'label': '1000', 'style': {'color': '#f50'}}
            },
    ),

    # Infection Rate input, slider
    "Infection rate: ",    
    dcc.Input(id="rate_input", type="number", min=.0001, max=1, value=5e-1, placeholder="Infection rate", style={'marginRight':'10px'}),
    dcc.Slider(
        id='rate_slider',
        min=.0001,
        max=1,
        step=.01,
        value=5e-1,
        marks={
            .0001: {'label': '0.0001', 'style': {'color': '#77b0b1'}},
            1: {'label': '1.0', 'style': {'color': '#f50'}}
            },
    ),          # Infection Rate Confidence slider
    html.Div(id='slider-output-container22'),
    dcc.Slider(
        id='rate-confidence-slider',
        min=80,
        max=100,
        step=1,
        value=100,
        marks={
            80: {'label': '80', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            },
        # tooltip={"placement": "top", "always_visible": True}
    )]
    ),
        
        
    html.Div(style={'columnCount': 1}, children=[

    # Recovery input, slider
    html.Div(id='slider-output-container3'),
    dcc.Input(id="recovery-input", type="number", min=.01, max=1, placeholder="Recovery rate", style={'marginRight':'10px'}),
    dcc.Slider(
        id='recovery-slider',
        min=.01,
        max=1,
        step=.01,
        value=.1,
        marks={
            .01: {'label': '0.01', 'style': {'color': '#77b0b1'}},
            1: {'label': '1.0', 'style': {'color': '#f50'}}
            },
    ),          #  Recovery confidence slider
    html.Div(id='slider-output-container33'),
    dcc.Slider(
        id='recovery-confidence-slider',
        min=80,
        max=100,
        step=1,
        value=100,
        marks={
            80: {'label': '80', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            }, 
    ),
    
    # Imunity input, slider
    html.Div(id='slider-output-container4'),
    dcc.Input(id="immunity-input", type="number", min=.01, max=1,  placeholder="Immunity", style={'marginRight':'10px'}),
    dcc.Slider(
        id='immunity-slider',
        min=.01,
        max=1,
        step=.01,
        value=.1,
        marks={
            .01: {'label': '0.01', 'style': {'color': '#77b0b1'}},
            1: {'label': '1.0', 'style': {'color': '#f50'}}
            },
    ),      # Imunity confidence slider
    html.Div(id='slider-output-container44'),
    dcc.Slider(
        id='immunity-confidence-slider',
        min=80,
        max=100,
        step=1,
        value=100,
        marks={
            80: {'label': '80', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            },
    )]
    ),
    
    
    html.Div(children=[
     
    # Vaccinated input, slider 
    html.Div(id='slider-output-container5'),
    dcc.Input(id="vaccinated-input", type="number", min=0, max=100, size="50", placeholder="Percent vaccinated"),
    dcc.Slider(
        id='vaccinated-slider',
        min=0,
        max=100,
        step=.1,
        value=0,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            },
    ),
    
    # Mask input, slider 
    html.Div(id='slider-output-container6'),
    dcc.Input(id="mask-input", type="number", min=0, max=100,  placeholder="Percent wearing masks", style={'marginRight':'10px'}),
    dcc.Slider(
        id='mask-slider',
        min=0,
        max=100,
        step=.1,
        value=0,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            },
        
    )]
        
    ),
    
    ])
    
    ])

# Update graph: Add log/lin buttons, size, color, legend, axis
plotly_fig.update_layout(
                    updatemenus=[dict(buttons=[
                    dict(
                        label="Linear",  
                        method="relayout", 
                        args=[{"yaxis.type": "linear"}]),
                    dict(
                        label="Log", 
                        method="relayout", 
                        args=[{"yaxis.type": "log"}])])],
                    
                    width=1350, height=500, paper_bgcolor='#222222',
                    plot_bgcolor ='#e3e3e3', xaxis_title="Days", 
                    yaxis_title="Population", 
                    font=dict(
                        size=16, color="#ffffff")      
        )


# Population input, slider sync
@app.callback(
    Output('pop_slider', 'value'),
    Output('pop_input', 'value'),
    Input('pop_slider', 'value'),
    Input('pop_input', 'value'))
def update_output0(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "pop_slider" :
        range_slider = s_value
    else:
        range_slider = i_value
         
    if trigger_id == "pop_input" :
          range_input = i_value 
    else :
        range_input = s_value
    
    return range_slider, range_input
        
# Range(days) input, slider sync
@app.callback(
    Output('range_slider', 'value'),
    Output('range_input', 'value'),
    Input('range_slider', 'value'),
    Input('range_input', 'value'))
def update_output1(s_value, i_value):
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

# Infection rate input, slider sync
@app.callback(
    Output('rate_slider', 'value'),
    Output('rate_input', 'value'),
    Input('rate_slider', 'value'),
    Input('rate_input', 'value'))
def update_output2(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "rate_slider" :
        range_slider = s_value
    else:
        range_slider = i_value
         
    if trigger_id == "rate_input" :
          range_input = i_value 
    else :
        range_input = s_value
    
    return range_slider, range_input


@app.callback(
    Output('slider-output-container3', 'children'),
    Input('recovery-slider', 'value'))
def update_output3(value):
    return 'Recovery rate: "{}"'.format(value)
@app.callback(
    Output('slider-output-container33', 'children'),
    Input('recovery-confidence-slider', 'value'))
def update_output333(value):
    return 'Confidence: "{}"'.format(value)

@app.callback(
    Output('slider-output-container4', 'children'),
    Input('immunity-slider', 'value'))
def update_output4(value):
    return 'Post infection immunity: "{}"'.format(value)
@app.callback(
    Output('slider-output-container44', 'children'),
    Input('immunity-confidence-slider', 'value'))
def update_output444(value):
    return 'Confidence: "{}"'.format(value)

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
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)

# def download_data(n_clicks):
#     return dcc.send_data_frame(plotly_fig.to_csv, "mydf.csv")




# Update Graph callback, interaction
@app.callback(
    Output('SIR-graph', 'figure'),
    Input('pop_slider', 'value'),
    Input('range_input', 'value'),
    Input('rate_slider', 'value'),
    Input('recovery-slider', 'value'),
    Input('immunity-slider', 'value'),
    Input('vaccinated-slider', 'value'),
    Input('mask-slider', 'value'),
    Input('rate-confidence-slider', 'value'),
    Input('recovery-confidence-slider', 'value'),
    # Input('imunity-confidence-slider', 'value'),
    )

def update_graph(pop_value, range_value, rate_value, recovery_value, 
                 immunity_value, vaccinated_value, mask_value, rate_confidence_value,
                 recovery_confidence_value):
    
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
    ax.plot(solution.t, solution.y[1]*rate_confidence_value/100, label = 'I (Lower bound)')
    ax.plot(solution.t, solution.y[1]*(abs((1-rate_confidence_value/100))+1), label = 'I (Upper Bound)')
    ax.plot(solution.t, solution.y[2], label = 'R')
    ax.plot(solution.t, solution.y[2]*recovery_confidence_value/100, label = 'R (Lower bound)')
    ax.plot(solution.t, solution.y[2]*(abs((1-recovery_confidence_value/100))+1), label = 'R (Upper bound)')
    ax.grid(True)
    ax.legend(["Susceptible", "Infected", "Recovered"])

    plotly_fig = mpl_to_plotly(fig)
    
    
    plotly_fig.update_layout(
                    updatemenus=[dict(buttons=[
                    dict(
                        label="Linear",  
                        method="relayout", 
                        args=[{"yaxis.type": "linear"}]),
                    dict(
                        label="Log", 
                        method="relayout", 
                        args=[{"yaxis.type": "log"}])])],
                    
                    width=1350, height=500, paper_bgcolor='#222222',
                    plot_bgcolor ='#e3e3e3', xaxis_title="Days", 
                    yaxis_title="Population", 
                    font=dict(
                        size=16, color="#ffffff")  
        )

    return plotly_fig



if __name__ == '__main__':
   app.run_server(debug = True)
    
    
    
    
