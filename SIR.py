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
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use("bmh")
from scipy.integrate import solve_ivp
import dash 
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from flask import request, Response


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
            
    html.H1(
    # children='COVID-19 Model', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 30}
    ),
    html.Div(children = 'COVID-19 Model', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 40, 'padding': "1px"}),
    
    dcc.Graph(id= 'SIR-graph', figure=plotly_fig),
    
    html.Div(style={'padding': 15, 'flex': 1},children=[
    html.A(
    "Download CSV",
    id="download_csv",
    href="#",
    className="btn btn-outline-secondary btn-sm",

    style={'textAlign': 'center', 'color': '#ffffff', 'padding': '6px', 'fontSize': 15, 'borderColor':"#ffffff"},
    
    )
    
    ])
    
    ]),

    
    
    html.Div(style={'display': 'flex', 'flex-direction': 'row'}, children=[
        html.Div(style={'padding': 15, 'flex': 1},children=[
            
    # Population input, slider
    "Population: ",
    dcc.Input(debounce=True, id="pop_input", type="number", min=1000, max=10000000,  value=100000, placeholder="Population"),
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
    dcc.Input(debounce=True, id="range_input", type="number", min=10, max=1000, value=200, placeholder="Days", style={'marginRight':'10px'}),
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
    dcc.Input(debounce=True, id="rate_input", type="number", min=.0001, max=1, value=5e-1, placeholder="Infection rate", style={'marginRight':'10px'}),
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
        tooltip={"placement": "top"},
        marks={
            80: {'label': '80', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            },
        # tooltip={"placement": "top", "always_visible": True}
    )]
    ),
        
        
    html.Div(style={'padding': 10, 'flex': 1}, children=[

    # Recovery input, slider
    "Recovery rate: ", 
    dcc.Input(debounce=True, id="recovery_input", type="number", min=.01, max=1, value=.1, placeholder="Recovery rate", style={'marginRight':'10px'}),
    dcc.Slider(
        id='recovery_slider',
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
        tooltip={"placement": "top"},
        marks={
            80: {'label': '80', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            }, 
    ),
    
    # Imunity input, slider
    "Post infection immunity: ",
    dcc.Input(debounce=True, id="immunity_input", type="number", min=.01, max=1,  value=.1, placeholder="Immunity", style={'marginRight':'10px'}),
    dcc.Slider(
        id='immunity_slider',
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
        tooltip={"placement": "top"},
        marks={
            80: {'label': '80', 'style': {'color': '#77b0b1'}},
            100: {'label': '100', 'style': {'color': '#f50'}}
            },
    )]
    ),
    
    
    html.Div(style={'padding': 10, 'flex': 1},children=[
     
    # Vaccinated input, slider 
    "Percent vaccinated: ",
    dcc.Input(debounce=True, id="vaccinated_input", type="number", min=0, max=100, value=0, size="50", placeholder="% vaccinated"),
    dcc.Slider(
        id='vaccinated_slider',
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
    "Percent wearing masks in public: ",
    dcc.Input(debounce=True, id="mask_input", type="number", min=0, max=100,  value=0, placeholder="% wearing masks", style={'marginRight':'10px'}),
    dcc.Slider(
        id='mask_slider',
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

# Update graph initially: Add log/lin buttons, size, color, legend, axis
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
    return 'Confidence: {}'.format(value), '%'

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
    return 'Confidence: {}'.format(value), '%'

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
    return 'Confidence: {}'.format(value), '%'

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




# Update Graph callback, interaction
@app.callback(
    Output('SIR-graph', 'figure'),
    # Output('download_csv', 'n_clicks'),
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
    # Input('some_input', 'value')
    )

def update_graph(pop_value, range_value, rate_value, recovery_value, 
                 immunity_value, vaccinated_value, mask_value, rate_confidence_value,
                 recovery_confidence_value, immunity_confidence_value):
    
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
    
    ax.plot(solution.t, solution.y[0], label = 'S', linewidth=3, color='#005d8f')
    ax.plot(solution.t, solution.y[1], label = 'I', linewidth=3, color='#8f71eb')
    ax.plot(solution.t, solution.y[2], label = 'R', linewidth=3, color='#de4e4e')
    
   
    rand_beta = np.random.triangular(beta_0*rate_confidence_value/100*.9999999999999, beta_0, beta_0*((1-(rate_confidence_value/100))+1)*1.0000000000001, 10)
    rand_gamma = np.random.triangular(gamma*recovery_confidence_value/100*.9999999999999, gamma, gamma*((1-(recovery_confidence_value/100))+1)*1.000000000001, 10)
    rand_epsilon = np.random.triangular(epsilon*immunity_confidence_value/100*.9999999999999, epsilon, epsilon*((1-(immunity_confidence_value/100))+1)*1.0000000000001, 10)
    
    for a, b, c in zip(rand_beta, rand_gamma, rand_epsilon):
        rand_solution = solve_ivp(sir_derivs, time_range, initial_conditions, t_eval=fine_time, args = (a, omega, b, c, N))
        ax.plot(solution.t, rand_solution.y[0], color='#80aec7', linewidth=3)
        ax.plot(solution.t, rand_solution.y[1], color='#bcaaf3', linewidth=3)
        ax.plot(solution.t, rand_solution.y[2], color='#eb9595', linewidth=3)
        

    ax.plot(solution.t, solution.y[0], label = 'S', linewidth=3, color='#005d8f')
    ax.plot(solution.t, solution.y[1], label = 'I', linewidth=3, color='#8f71eb')
    ax.plot(solution.t, solution.y[2], label = 'R', linewidth=3, color='#de4e4e')
   
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
    
    
     
    df = pd.DataFrame({
     'S':[solution.t, solution.y[0]],
     'I':[solution.t, solution.y[1]],
     'R':[solution.t, solution.y[2]]})
    
    return plotly_fig


# @app.callback(
#     Output('download_csv', 'href'),
#     [Input('some_input', 'value')]
# )

# def some_callback(input_value):
#     """Some callback that updates the href for the button"""
    
#     return f"/download_csv?value={input_value}"

# @app.route('/download_csv')
# def dash_download_csv():
#         """Regular Flask route.
#         Download a CSV file from an existing Pandas DataFrame"""

#         # Here's the argument passed to the URL in the Dash callback
#         value = request.args.get('value')
#         df = get_df(value)
    
#         # Convert DataFrame to CSV
#         csv = df.to_csv(index=False)

#         return Response(
#             csv,
#             mimetype="text/csv",
#             headers={
#             "Content-disposition": "attachment; filename=rcom_data.csv"
#         }
#     )





if __name__ == '__main__':
    app.run_server(debug = True)
    
    
    
    
