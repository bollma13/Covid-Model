#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:38:29 2022

@author: sambollman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
from scipy.integrate import solve_ivp
import dash 
import dash_html_components as html
import dash_core_components as dcc
from plotly.tools import mpl_to_plotly
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])
server = app.server

cell_type = "code"
execution_count = 2


# New York State
k1 = 1/5.2                  # The mean incubation time (days)
k2 = 1/10                   # The mean time from mild/moderate stage to severe/critical stage (days)
r = 4                       # The mean number of members in a family
gamma1 = 1/7                # The average recovery period for diagnosed mild/moderate cases (days)
gamma0 = 1/9.6620           # The mean time for natural recovery (days)
alpha1 = 1/6.9974           # The average period from symptoms onset to diagnose for mild/moderate cases (days)
alpha2 = 1/2.0421           # The average diagnose period for severe/critical cases (days)
gamma2 = 1/(21 - 1/alpha2)  # The average recovery period for diagnosed severe/critical cases
xi = 1/28                   # The mean recovery periodfor infected family members (days)

N0 = 19_542_209             # Initial population

# initial values
V0 = 0                          # The initial value of vaccinated individuals
Nf0 = 4                         # The initial value of households susceptible to COVID
Sf0 = 3                         # The initial value of suseptable household members that are susceptible to COVID
Vf0 = 0                         # The initial value of vacinated household members
E0 = 99.9998                # The initial value of latent individuals
A0 = 10                     # The initial value of asymptomatic individuals
I10 = 10                    # The initial value of undiagnosed mild/moderate cases
I20 = 100                   # The initial value of undiagnosed severe/critical individuals
T10 = 0                         # The initial value of diagnosed mild/moderate cases
T20 = 0                         # The initial value of diagnosed severe/critical individuals
R0 = 0                          # The initial value of recovered individuals
D0 = 0                          # The initial value of dead individuals
S0 = N0 - V0 - E0 - A0 - I10 - I20 - T10 - T20 - R0    # The initial value of susceptible individuals

beta = .0305                # The per-act transmission probability in contact withinfected individuals with symptoms
epsilon = .75               # The reduction in per-act transmission probability ifinfection is in latent and asymptomatic stage

rho = .5092                 # The probability that an individual is asymptomatic
k3 = .0123*k2               # The progression rate from diagnosed mild/moderate stage to diagnosed severe/critical stage

# contacts
m_ini = 42.0                # Base daily contact number in the public settings
m0 = .7966                  # Change rate of daily contact number
t_ini = 28.0137             # The time when the contact number is half of maximal and minimal contact number in public settings  (before reopening)
t_reopen = 100              # The time when the contact number is half of maximal and minimal contact number in public settings (after reopening)
 
# hand washing
q_ini = .77                 # Base percentage of handwashing before the epidemic
qbar = .95                  # Maximal percentage of handwashing during the epidemic
theta2 = .42                # The effectiveness of handwashing in preventing infection

# face masks
p_ini = 0                   # Base percentage of face mask usage in the public settings before the Executive Order on face mask use
pbar = .766                 # Percentage of face mask usage in the public settings after the Executive Order on face mask use
tp = 42                     # The time when face mask usage in the public settings is half of the maximal face mask usage rate
p2 = 0                      # The usage percentage of mask in the households
theta1 = .85                # The effectiveness of mask in preventing infection

mu1 = .05                   # Disease-induced death rate of undiagnosed severe/critical cases
mu2 = .1131*mu1             # Disease-induced death rate of diagnosed severe/critical cases

# vaccines
w = 0               # vaccination rate 
ev = 1              # vaccine effectivness 
 
def m1(t,m_ini,t_ini,t_reopen,m0): 
    if t < t_reopen: 
        return m_ini + ( (.2*m_ini - m_ini)/( 1 + np.exp(-m0*(t-t_ini)) ) ) # before reopening 
    else: 
        return .2*m_ini + ( (3.3812 * .2*m_ini - .2*m_ini)/( 1 + np.exp(-m0*(t-t_reopen)) ) ) # after reopen 
 
def m2(t,t_ini,t_reopen,m0): 
    if t < t_reopen: 
        return 4 + ( 4/( 1 + np.exp(-m0*(t - t_ini)) ) ) 
    else: 
        return 8 + ( (.8*8 - 8)/(1 + np.exp(-m0*(t - t_reopen))) ) 
     
def q(t,qbar,q_ini,m0,t_ini): 
    return q_ini + ( (qbar-q_ini)/(1 + np.exp(-m0*(t-t_ini))) ) 
 
def p1(t,p_ini,pbar,tp): 
    return p_ini + ( (pbar-p_ini)/(1 + np.exp(-(t-tp))) )
 


def covid_model(t,y,k1,k2,r,gamma1,gamma0,alpha1,alpha2,gamma2,xi,beta,epsilon,rho,k3,m_ini,m0,t_ini,t_reopen,q_ini,qbar,theta2,p_ini,pbar,tp,p2,theta1,mu1,mu2,w, ev): 
      
#   N = np.sum(y) - y[-1] - y[-2] 

    N = y[0] + y[1] + y[5] + y[6] + y[7] + y[8] + y[9] + y[10] + y[11] 
      
    betaI_pub = beta*m1(t,m_ini,t_ini,t_reopen,m0)*(1-theta1*p1(t,p_ini,pbar,tp))*(1-theta2*q(t,qbar,q_ini,m0,t_ini)) 
    betaE_pub = (1-epsilon)*betaI_pub 
     
    betaI_pri = beta*m2(t,t_ini,t_reopen,m0)*(1-theta1*p2)*(1-theta2*q(t,qbar,q_ini,m0,t_ini)) 
    betaE_pri = (1-epsilon)*betaI_pri 
     
    gamma_pub = betaE_pub * ( ((y[0]-y[3])*(y[5]+y[6]))/(N-y[2]) ) + betaI_pub * ( ((y[0]-y[3])*(y[7]+y[8]))/(N-y[2]) ) 
    gamma_pri = betaE_pri * ( (y[3]*(y[5]+y[6]))/y[2] ) + betaI_pri * ( (y[3]*(y[7]+y[8]))/y[2] ) 
     
    gammaV_pub = betaE_pub * (1 - ev) * ( ((y[1]-y[4])*(y[5]+y[6]))/(N-y[2]) ) + betaI_pub*(1-ev)*( ((y[1]-y[4])*(y[7]+y[8]))/(N-y[2]) ) 
    gammaV_pri = betaE_pri * (1 - ev) * ( (y[4]*(y[5]+y[6]))/y[2] ) + betaI_pri * (1-ev)*( (y[4]*(y[7]+y[8]))/y[2] ) 
     
    sdot = -gamma_pub - gamma_pri - w*y[0] 
    vdot = w*y[0] - gammaV_pub - gammaV_pri 
    nfdot = r*(gamma_pub + gammaV_pub) - xi*y[2] 
    sfdot = (r-1)*(gamma_pub + gammaV_pub) * (y[3]/(y[3] + y[4])) - gamma_pri - xi*y[3] 
    vfdot = (r-1)*(gamma_pub + gammaV_pub) * (y[4]/(y[3] + y[4])) - gammaV_pri - xi*y[4] 
    edot = gamma_pub + gamma_pri + gammaV_pub + gammaV_pri - k1*y[5] 
    adot = k1*rho*y[5] - gamma0*y[6] 
    I1dot = k1*(1-rho)*y[5] - alpha1*y[7] - k2*y[7] - gamma0*y[7] 
    I2dot = k2*y[7] - alpha2*y[8] - mu1*y[8] 
    T1dot = alpha1*y[7] - gamma1*y[9] - k3*y[9] 
    T2dot = alpha2*y[8] + k3*y[9] - gamma2*y[10] - mu2*y[10] 
    Rdot = gamma0*y[6] + gamma0*y[7] + gamma1*y[9] + gamma2*y[10] 
    Ddot = mu1*y[8] + mu2*y[10] 
    Cdot = alpha1*y[7] + alpha2*y[8] 
     
    return [sdot, vdot, nfdot, sfdot, vfdot, edot, adot, I1dot, I2dot, T1dot, T2dot, Rdot, Ddot, Cdot]


  # day 0 = March 1 
  # day 245 = Oct 31 
res = solve_ivp(covid_model, [0,244], [S0, V0, Nf0, Sf0, Vf0, E0, A0, I10, I20, T10, T20, R0, D0, 110], args=[k1,k2,r,gamma1,gamma0,alpha1,alpha2,gamma2,xi,beta,epsilon,rho,k3,m_ini,m0,t_ini,t_reopen,q_ini,qbar,theta2,p_ini,pbar,tp,p2,theta1,mu1,mu2,w, ev] ,dense_output=True)


fig = plt.figure(figsize=(16,10)) 
tt = np.linspace(0,244,245) 


plt.plot(tt,res.sol(tt)[0],label='S') 
# plt.plot(tt,res.sol(tt)[1],label='V') 
# plt.plot(tt,res.sol(tt)[2],label=r'$N_f$') 
# plt.plot(tt,res.sol(tt)[3],label=r'$S_f$') 
# plt.plot(tt,res.sol(tt)[4],label=r'$V_f$') 
# plt.plot(tt,res.sol(tt)[5],label='E') 
# plt.plot(tt,res.sol(tt)[6],label='A') 
# plt.plot(tt,res.sol(tt)[7],label=r'$I_1$') 
# plt.plot(tt,res.sol(tt)[8],label=r'$I_2$') 
# plt.plot(tt,res.sol(tt)[9],label=r'$T_1$') 
# plt.plot(tt,res.sol(tt)[10],label=r'$T_2$') 
plt.plot(tt,res.sol(tt)[11],label='R') 
plt.plot(tt,res.sol(tt)[12],label='D') 
# plt.plot(tt,res.sol(tt)[13],label='cases') 
 
# plt.ylim(0,2e5) 
plt.xlim(0,244) 
plt.grid()  
plt.legend()


fig = plt.Figure()
ax = fig.add_subplot(111)

ax.plot(tt,res.sol(tt)[0],label='susceptible', color='#005d8f')
ax.plot(tt,res.sol(tt)[1],label='vaccinated') 
ax.plot(tt,res.sol(tt)[5],label='latent infections') 
ax.plot(tt,res.sol(tt)[6],label='asymptomatic infections') 
ax.plot(tt,res.sol(tt)[7]+res.sol(tt)[8],label='undiagnosed infections') 
ax.plot(tt,res.sol(tt)[9]+res.sol(tt)[10],label='diagnosed infections') 
ax.plot(tt,res.sol(tt)[11],label='recovered', color='#8f71eb') 
ax.plot(tt,res.sol(tt)[12],label='dead', color='#de4e4e') 

plt.xlabel("time")
ax.grid(True)

plotly_fig = mpl_to_plotly(fig)


N0 = 19_542_209             # Initial population

# initial values
V0 = 0                          # The initial value of vaccinated individuals
Nf0 = 4                         # The initial value of households susceptible 
Sf0 = 3                         # The initial value of household members that are susceptible
Vf0 = 0                         # The initial value of vacinated household members
E0 = 99.9998                # The initial value of latent individuals
A0 = 10                     # The initial value of asymptomatic individuals
I10 = 10                    # The initial value of undiagnosed mild/moderate cases
I20 = 100                   # The initial value of undiagnosed severe/critical individuals
T10 = 0                         # The initial value of diagnosed mild/moderate cases
T20 = 0                         # The initial value of diagnosed severe/critical individuals
R0 = 0                          # The initial value of recovered individuals
D0 = 0                          # The initial value of dead individuals
S0 = N0 - V0 - E0 - A0 - I10 - I20 - T10 - T20 - R0    # The initial value of susceptible individuals

datatable = {"Variable": ["V0", "Nf0", "Sf0", "Vf0", "T10", "T20", "R0", "D0", "S0", "k1", "k2", "r", "gamma1", "gamma0", "alpha1", "alpha2", "gamma2", "xi", "E0", "A0", "A0","A0","beta","epsilon","rho","k3","m_ini","m0",
                                  "t_ini","t_reopen","q_ini","qbar","theta2","p_ini","pbar","tp","p2","theta1","mu1","mu2","w","ev"]
                    ,"Description": ["initial value of vaccinated individuals", "initial value of households susceptible ", "initial value of household members that are susceptible", "initial value of vacinated household members", "initial value of diagnosed mild/moderate cases", "initial value of diagnosed severe/critical individuals", "initial value of recovered individuals", "initial value of dead individuals", "initial value of susceptible individuals",  "mean incubation time (days)", "mean time from mild/moderate stage to severe/critical stage (days)", "mean number of members in a family", "average recovery period for diagnosed mild/moderate cases (days)", "mean time for natural recovery (days)", "average period from symptoms onset to diagnose for mild/moderate cases (days)",
                                     "average diagnose period for severe/critical cases (days)","average recovery period for diagnosed severe/critical cases (days)","the mean recovery periodfor infected family members (days)","the initial value of latent individuals","the initial value of asymptomatic individuals","the initial value of undiagnosed mild/moderate cases",
                                     "initial value of undiagnosed severe/critical individuals","per-act transmission probability in contact withinfected individuals with symptoms","reduction in per-act transmission probability ifinfection is in latent and asymptomatic stage","probability that an individual is asymptomatic",
                                     "progression rate from diagnosed mild/moderate stage to diagnosed severe/critical stage","base daily contact number in the public settings","change rate of daily contact number","time when the contact number is half of maximal and minimal contact number in public settings (before reopening)","time when the contact number is half of maximal and minimal contact number in public settings (after reopening)",
                                     "base percentage of handwashing before the epidemic","maximal percentage of handwashing during the epidemic","effectiveness of handwashing in preventing infection","base percentage of face mask usage in the public settings before the Executive Order on face mask use","percentage of face mask usage in the public settings after the Executive Order on face mask use","time when face mask usage in the public settings is half of the maximal face mask usage rate",
                                     "usage percentage of mask in the households","effectiveness of mask in preventing infection","disease-induced death rate of undiagnosed severe/critical cases","disease-induced death rate of diagnosed severe/critical cases","vaccination rate","vaccine effectivness"]
                 }

data = pd.DataFrame(datatable)
 

app.layout = html.Div([
        html.Div(children = 'COVID-19 Model', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 40, 'padding': "5px"}),
        dcc.Tabs([

    dcc.Tab(label='Controls', 
        children=[
         
        html.Div(style={'textAlign': 'center','display': 'flex', 'flex-direction': 'row'}, children=[
        
        html.Div(style={'padding': 10, 'flex': 1},children=[
            
        
        html.Div(children = 'Quick Start', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
        html.Label('Use the text boxes or sliders to edit values on the right. Click the \'Plots\' tab at the top to see the result. For more details click on the \'More Info\' tab.', 
                   style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px"}),
        html.Div(children = 'Compare', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "15px"}),
        html.Div(children = 'note: toggle updates other graph controls immediately', style={'textAlign': 'center', 'color': '#b6e0f8', 'marginBottom' : '10px'}),

        html.Div([
                daq.ToggleSwitch(label='', color='gray',
                                 labelPosition='bottom',
                    id='compare_toggle',
                    value=False
                    ),
                html.Div(id='compare_toggle_output', style={'textAlign': 'center', 'color': '#bbbbbb', 'marginTop': '10px'})
                ]),  
        html.Div(style={'padding': 15, 'flex': 1},children=[
        html.Label(children = '', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px"})
        
            ])
        
    ]),
         
    html.Div(style={'textAlign': 'center','padding': 15, 'flex': 1},children=[
        
     # Range input, slider
     html.Div(children = 'days', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="range_input", type="number", min=10, max=600,  value=244, placeholder="Range", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='range_slider',
        min=10,
        max=600,
        step=1,
        value=244,
        marks={
            10: {'label': '10', 'style': {'color': '#999999', 'fontSize': 15}},
            600: {'label': '600', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),  
        
    html.Button('Initial Conditions', disabled='disabled',
                style= {'background-color': '#616161',
                      'color': 'white',
                      'fontSize': 18,
                      'height': '1.8%',
                      'width': '90%',
                      'marginTop' : '10px'
                      }),

    
    #   V0 -   The initial value of vaccinated individuals      input, slider 
    html.Div(children = 'initial value of vaccinated individuals (V0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="V0_input", type="number", min= 0 , max= 1000000 ,  value= 0 , placeholder="0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='V0_slider',
        min= 0 ,
        max= 1000000 ,
        step= 100,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            1000000: {'label': '1000000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #  Nf0 -  The initial value of households susceptible to COVID    input, slider 
    html.Div(children = 'initial value of households susceptible to COVID (Nf0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="Nf0_input", type="number", min= 0 , max= 10000 ,  value= 4 , placeholder="0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='Nf0_slider',
        min= 0 ,
        max= 10000 ,
        step= 1,
        value= 4 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            10000: {'label': '10000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   Sf0 -   The initial value of  household members that are susceptible to COVID    input, slider 
    html.Div(children = 'initial value of household members that are susceptible to COVID (Sf0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="Sf0_input", type="number", min= 0 , max= 40000 ,  value= 3 , placeholder="Sf0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='Sf0_slider',
        min= 0 ,
        max= 0 ,
        step= 1,
        value= 3 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            40000: {'label': '40000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   Vf0 -   The initial value of   vacinated household members   input, slider 
    html.Div(children = 'initial value of vacinated household members (Vf0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="Vf0_input", type="number", min= 0 , max= 1000000 ,  value= 0 , placeholder="Vf0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='Vf0_slider',
        min= 0 ,
        max= 0 ,
        step=1 ,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            100000: {'label': '1000000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    
          
    # E0 = 99.9998                # The initial value of latent individuals
    html.Div(children = 'initial value of latent individuals (E0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="E0_input", type="number", min=95, max=105,  value=99.9998, placeholder="E0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='E0_slider',
        min=95,
        max=105,
        step=.1,
        value= 99.9998,
        marks={
            95: {'label': '95', 'style': {'color': '#999999', 'fontSize': 15}},
            105: {'label': '105', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   A0 -   The initial value of asymptomatic individuals     input, slider 
    html.Div(children = 'initial value of asymptomatic individuals (A0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="A0_input", type="number", min= 8 , max= 12 ,  value= 10 , placeholder="A0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='A0_slider',
        min= 8 ,
        max= 12 ,
        step=.1,
        value= 10 ,
        marks={
            8: {'label': '8', 'style': {'color': '#999999', 'fontSize': 15}},
            12: {'label': '12', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    
     #  I10 - The initial value of undiagnosed mild/moderate cases      input, slider 
    html.Div(children = 'initial value of undiagnosed mild/moderate (I10)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="I10_input", type="number", min= 8 , max= 12 ,  value= 10 , placeholder="I10", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='I10_slider',
        min= 8 ,
        max= 12 ,
        step=.1,
        value= 10 ,
        marks={
            8: {'label': '8', 'style': {'color': '#999999', 'fontSize': 15}},
           12 : {'label': '12', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
     #   I20 - The initial value of undiagnosed severe/critical individuals input, slider 
    html.Div(children = 'initial value of undiagnosed severe/critical (I20)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="I20_input", type="number", min=  90, max=  110,  value= 100 , placeholder="I20", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='I20_slider',
        min=  90,
        max=  110,
        step=.1,
        value= 100 ,
        marks={
            90: {'label': '90', 'style': {'color': '#999999', 'fontSize': 15}},
            110: {'label': '110', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
     #   T10 -   The initial value of   diagnosed mild/moderate individuals   input, slider 
    html.Div(children = 'initial value of diagnosed mild/moderate cases (T10)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="T10_input", type="number", min= 0 , max= 10000 ,  value= 0 , placeholder="T10", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='T10_slider',
        min= 0 ,
        max= 10000 ,
        step= 1,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            10000: {'label': '10000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   T20 -   The initial value of   diagnosed severe/critical individuals   input, slider 
    html.Div(children = 'initial value of diagnosed severe/critical individuals (T20)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="T20_input", type="number", min= 0 , max= 10000 ,  value= 0 , placeholder="T20", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='T20_slider',
        min= 0 ,
        max= 10000 ,
        step= 1,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            10000: {'label': '10000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   R0 -   The initial value of   recovered individuals   input, slider 
    html.Div(children = 'initial value of recovered individuals (R0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="R0_input", type="number", min= 0 , max= 10000000 ,  value= 0 , placeholder="R0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='R0_slider',
        min= 0 ,
        max= 10000000 ,
        step= 1,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            10000000: {'label': '10000000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   D0 -   The initial value of   dead individuals   input, slider 
    html.Div(children = 'initial value of dead individuals (D0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="D0_input", type="number", min= 0 , max= 100000 ,  value= 0 , placeholder="D0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='D0_slider',
        min= 0 ,
        max= 100000 ,
        step= 1,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            100000: {'label': '100000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    #   C0 -   The initial value of   confirmed cases   input, slider 
    html.Div(children = 'initial value of confirmed cases (C0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="C0_input", type="number", min= 0 , max= 100000 ,  value= 0 , placeholder="C0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='C0_slider',
        min= 0 ,
        max= 100000 ,
        step= 1,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            100000: {'label': '100000', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
     
    
    ]),
    
    html.Div(style={'padding': 15, 'flex': 1}, children=[
    html.Button('Disease', disabled='disabled',
                style= {'background-color': '#616161',
                      'color': 'white',
                      'fontSize': 18,
                      'height': '1.8%',
                      'width': '90%',
                      'marginTop' : '10px'
                      }),
     # k1 input, slider
    html.Div(children = 'incubation time (k1)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="k1_input", type="number", min=4.1, max=7,  value=5.2, placeholder="k1", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='k1_slider',
        min=4.1,
        max=7,
        step=.1,
        value=5.2,
        marks={
            4.1: {'label': '4.1', 'style': {'color': '#999999', 'fontSize': 15}},
            7: {'label': '7', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    # k2 mild/moderate stage to severe/critical stage (k2) input, slider
    html.Div(children = 'mild/moderate stage to severe/critical stage (k2)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="k2_input", type="number", min=8, max=12, value=10, placeholder="k2", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='k2_slider',
        min=8,
        max=12,
        step=.1,
        value=10,
        marks={
            8: {'label': '8', 'style': {'color': '#999999', 'fontSize': 15}},
            12: {'label': '12', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    # gamma1 - The average recovery period for diagnosed mild/moderate cases (days) input, slider 
    html.Div(children = 'recover period for mild/moderate (gamma1)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="gamma1_input", type="number", min=5.5, max=8.5,  value=7, placeholder="gamma1", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='gamma1_slider',
        min=5.5,
        max=8.5,
        step=.1,
        value=7,
        marks={
            5.5: {'label': '5.5', 'style': {'color': '#999999', 'fontSize': 15}},
            8.5: {'label': '8.5', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    # gamma0 = 1/9.6620   # The mean time for natural recovery (days) input, slider 
    html.Div(children = 'natural recovery time (gamma0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="gamma0_input", type="number", min=8, max=11,  value=9.6620, placeholder="gamma0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='gamma0_slider',
        min=8,
        max=11,
        step=.1,
        value=9.6620,
        marks={
            8: {'label': '8', 'style': {'color': '#999999', 'fontSize': 15}},
            11: {'label': '11', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    # alpha1 = 1/6.9974  # The average period from symptoms onset to diagnose for mild/moderate cases (days) input, slider 
    html.Div(children = 'symptoms onset to diagnose for mild/moderate cases (alpha1)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="alpha1_input", type="number", min=6, max=8,  value=6.9974, placeholder="alpha1", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='alpha1_slider',
        min=6,
        max=8,
        step=.1,
        value=6.9974,
        marks={
            6: {'label': '6', 'style': {'color': '#999999', 'fontSize': 15}},
            8: {'label': '8', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    # alpha2 = 1/2.0421           # The average diagnose period for severe/critical cases (days) input, slider 
    html.Div(children = 'diagnose period for severe/critical cases (alpha2)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="alpha2_input", type="number", min=1, max=3,  value=2.0421, placeholder="alpha2", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='alpha2_slider',
        min=1,
        max=3,
        step=.1,
        value=2.0421,
        marks={
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}},
            3: {'label': '3', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    # gamma2 = 1/(21 - 1/alpha2)  # The average recovery period for diagnosed severe/critical cases input, slider 
    html.Div(children = 'recovery period for diagnosed severe/critical (gamma2)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="gamma2_input", type="number", min=18, max=24,  value=21, placeholder="gamma2", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='gamma2_slider',
        min=18,
        max=24,
        step=.1,
        value=21,
        marks={
            18: {'label': '18', 'style': {'color': '#999999', 'fontSize': 15}},
            24: {'label': '24', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),
    # xi = 1/28   # The mean recovery period for infected family members (days) input, slider 
    html.Div(children = 'recovery period for infected family members (xi)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="xi_input", type="number", min=24, max=32,  value=28, placeholder="xi", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='xi_slider',
        min=24,
        max=32,
        step=.1,
        value=28,
        marks={
            24: {'label': '24', 'style': {'color': '#999999', 'fontSize': 15}},
            28: {'label': '28', 'style': {'color': '#999999', 'fontSize': 15}}
            },     
    ),

   #   beta - The per-act transmission probability in contact withinfected individuals with symptoms input, slider 
    html.Div(children = 'per-act transmission probability (beta)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="beta_input", type="number", min= .02 , max= .04 ,  value= .0305 , placeholder="beta", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='beta_slider',
        min= .02 ,
        max= .04 ,
        step=.001,
        value= .0305 ,
        marks={
            .02: {'label': '.02', 'style': {'color': '#999999', 'fontSize': 15}},
            .04: {'label': '.04', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    
    #   epsilon - The reduction in per-act transmission probability ifinfection is in latent and asymptomatic stage     input, slider 
    html.Div(children = 'reduction in per-act transmission probability (epsilon)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="epsilon_input", type="number", min=  .65, max= .85 ,  value= .75 , placeholder="epsilon", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='epsilon_slider',
        min= .65 ,
        max= .85 ,
        step=.01,
        value= .75 ,
        marks={
            .65: {'label': '.65', 'style': {'color': '#999999', 'fontSize': 15}},
           .85 : {'label': '.85', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   rho - The probability that an individual is asymptomatic input, slider 
    html.Div(children = 'asymptomatic probability (rho)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="rho_input", type="number", min= .4 , max= .6 ,  value= .5092 , placeholder="rho", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='rho_slider',
        min= .4 ,
        max= .6 ,
        step=.01,
        value=  .5092,
        marks={
            .4: {'label': '.4', 'style': {'color': '#999999', 'fontSize': 15}},
            .6: {'label': '.6', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #      k3 - The progression rate from diagnosed mild/moderate stage to diagnosed severe/critical stage  input, slider 
    html.Div(children = 'progression rate from diagnosed mild/moderate stage to diagnosed severe/critical (k3)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="k3_input", type="number", min= .01 , max= .014 ,  value=  .0123, placeholder="k3", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='k3_slider',
        min=  .01,
        max=  .014,
        step=.001,
        value= .0123 ,
        marks={
            .01: {'label': '.01', 'style': {'color': '#999999', 'fontSize': 15}},
            .014: {'label': '.014', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #  m_ini - Base daily contact number in the public settings      input, slider 
    html.Div(children = 'daily contact number (m_ini)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="m_ini_input", type="number", min= 30 , max= 50 ,  value= 42 , placeholder="m_ini", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='m_ini_slider',
        min= 30 ,
        max= 50 ,
        step=1,
        value=  42,
        marks={
           30 : {'label': '30', 'style': {'color': '#999999', 'fontSize': 15}},
            50: {'label': '50', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #  m0 - Change rate of daily contact number      input, slider 
    html.Div(children = 'change rate of daily contact number (m0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="m0_input", type="number", min= .7 , max=  .9,  value= .7966 , placeholder="m0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='m0_slider',
        min= .7 ,
        max= .9 ,
        step=.01,
        value= .7966 ,
        marks={
           .7 : {'label': '.7', 'style': {'color': '#999999', 'fontSize': 15}},
            .9: {'label': '.9', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   t_ini - The time when the contact number is half of maximal and minimal contact number in public settings  (before reopening)     input, slider 
    html.Div(children = 'time when contact number is half of maximal and minimal contact number before reopening (t_ini)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="t_ini_input", type="number", min= 26 , max= 30 ,  value= 28.0137 , placeholder="t_ini", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='t_ini_slider',
        min= 26 ,
        max= 30 ,
        step=.1,
        value= 28.0137 ,
        marks={
           26 : {'label': '26', 'style': {'color': '#999999', 'fontSize': 15}},
            30: {'label': '30', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #  t_reopen - The time when the contact number is half of maximal and minimal contact number in public settings (after reopening)      input, slider 
    html.Div(children = 'time when contact number is half of maximal and minimal contact number after reopening (t_reopen)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="t_reopen_input", type="number", min=  90, max=  110,  value= 100 , placeholder="t_reopen", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='t_reopen_slider',
        min= 90 ,
        max= 110 ,
        step=.1,
        value=  100,
        marks={
            90: {'label': '90', 'style': {'color': '#999999', 'fontSize': 15}},
            110: {'label': '110', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    
    #    mu1 - Disease-induced death rate of undiagnosed severe/critical cases    input, slider 
    html.Div(children = 'death rate of undiagnosed severe/critical (mu1)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="mu1_input", type="number", min= .04 , max= .06 ,  value= .05  , placeholder="mu1", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='mu1_slider',
        min= .04 ,
        max= .06 ,
        step=.001,
        value= .05  ,
        marks={
            .04: {'label': '.04', 'style': {'color': '#999999', 'fontSize': 15}},
            .06: {'label': '.06', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   mu2 - Disease-induced death rate of diagnosed severe/critical cases     input, slider 
    html.Div(children = 'death rate of diagnosed severe/critical (mu2)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="mu2_input", type="number", min= .1 , max= .12 ,  value= .1131 , placeholder="mu2", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='mu2_slider',
        min= .1 ,
        max= .12 ,
        step=.01,
        value= .1131 ,
        marks={
            .1: {'label': '.1', 'style': {'color': '#999999', 'fontSize': 15}},
            .12: {'label': '.12', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    
    
    
    ]),
    html.Div(style={'textAlign': 'center','padding': 15, 'flex': 1},children=[
    html.Button('Demographics', disabled='disabled',
                style= {'background-color': '#616161',
                      'color': 'white',
                      'fontSize': 18,
                      'height': '1.8%',
                      'width': '90%',
                      'marginTop' : '10px'
                      }),
        
    # n0 population input, slider 
    html.Div(children = 'population (N0)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="N0_input", type="number", min=1000000, max=30000000, value=19542209, placeholder="N0", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='N0_slider',
        min=1000000,
        max=30000000,
        step=100,
        value=19542209,
        marks={
            1000000: {'label': '1000000', 'style': {'color': '#999999', 'fontSize': 15}},
            30000000: {'label': '30000000', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    
    # r family input, slider 
    html.Div(children = 'members in a family (r)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="r_input", type="number", min=2, max=6, value=4, placeholder="r", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='r_slider',
        min=2,
        max=6,
        step=.1,
        value=4,
        marks={
            2: {'label': '2', 'style': {'color': '#999999', 'fontSize': 15}},
            6: {'label': '6', 'style': {'color': '#999999', 'fontSize': 15}}
            },
    ),
    
    html.Button('Interventions', disabled='disabled',
                style= {'background-color': '#616161',
                      'color': 'white',
                      'fontSize': 18,
                      'height': '1.8%',
                      'width': '90%',
                      'marginTop' : '10px'
                      }),
        #   ev - vaccine effectivness     input, slider 
    html.Div(children = 'vaccine effectivness (ev)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="ev_input", type="number", min=  .8, max= 1 ,  value=  1, placeholder="ev", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='ev_slider',
        min= .8 ,
        max= 1 ,
        step=.01,
        value= 1 ,
        marks={
            .8: {'label': '.8', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #    w - vaccination rate     input, slider 
    html.Div(children = 'vaccination rate (w)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="w_input", type="number", min= 0 , max= .005 ,  value= 0 , placeholder="w", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='w_slider',
        min=  0,
        max= .005 ,
        step=.0001,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            .005: {'label': '.5', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #    q_ini - Base percentage of handwashing before the epidemic    input, slider 
    html.Div(children = 'handwashing percentage before the epidemic (q_ini)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="q_ini_input", type="number", min= .65 , max= .85 ,  value= .77 , placeholder="q_ini", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='q_ini_slider',
        min= .65 ,
        max= .85 ,
        step=.01,
        value= .77 ,
        marks={
            .65: {'label': '.65', 'style': {'color': '#999999', 'fontSize': 15}},
            .85: {'label': '.85', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   qbar - Maximal percentage of handwashing during the epidemic     input, slider 
    html.Div(children = 'handwashing percentage during the epidemic (qbar)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="qbar_input", type="number", min= .5 , max= 1 ,  value= .95 , placeholder="qbar", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='qbar_slider',
        min= .5 ,
        max= 1 ,
        step=.01,
        value= .95 ,
        marks={
            .5: {'label': '.5', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   theta2 = .42                # The effectiveness of handwashing in preventing infection     input, slider 
    html.Div(children = 'effectiveness of handwashing in preventing infection (theta2)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="theta2_input", type="number", min= 0 , max= .75 ,  value= .42 , placeholder="theta2", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='theta2_slider',
        min= 0 ,
        max= .75 ,
        step=.1,
        value= .42 ,
        marks={
           0 : {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            .75: {'label': '.75', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #    p_ini = 0                   # Base percentage of face mask usage in the public settings before the Executive Order on face mask use    input, slider 
    html.Div(children = 'face mask usage before the executive order (p_ini)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="p_ini_input", type="number", min=  0, max= .5 ,  value= 0 , placeholder="p_ini", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='p_ini_slider',
        min= 0 ,
        max= .5 ,
        step=.01,
        value= 0 ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            .5: {'label': '.5', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   pbar - Percentage of face mask usage in the public settings after the Executive Order on face mask use     input, slider 
    html.Div(children = 'face mask usage after the Executive Order (pbar)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="pbar_input", type="number", min= 0 , max= 1 ,  value= .766   , placeholder="pbar", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='pbar_slider',
        min= 0 ,
        max= 1 ,
        step=.01,
        value= .766   ,
        marks={
            0: {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #    tp - The time when face mask usage in the public settings is half of the maximal face mask usage rate    input, slider 
    html.Div(children = 'time when face mask usage is half of max (tp)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="tp_input", type="number", min= 30 , max= 55 ,  value= 42 , placeholder="tp", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='tp_slider',
        min= 30 ,
        max= 55 ,
        step=.1,
        value=42 ,
        marks={
            30: {'label': '30', 'style': {'color': '#999999', 'fontSize': 15}},
            55: {'label': '55', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   p2 - The usage percentage of mask in the households     input, slider 
    html.Div(children = 'percentage of masks in households (p2)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="p2_input", type="number", min= 0 , max= 1 ,  value= 0 , placeholder="p2", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='p2_slider',
        min= 0 ,
        max= 1 ,
        step=.01,
        value= 0 ,
        marks={
           0 : {'label': '0', 'style': {'color': '#999999', 'fontSize': 15}},
            1: {'label': '1', 'style': {'color': '#999999', 'fontSize': 15}}
            },  
    ),
    #   theta1 - The effectiveness of mask in preventing infection     input, slider 
    html.Div(children = 'effectiveness of masks (theta1)', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 18, 'padding': "15px"}),
    dcc.Input(debounce=True, id="theta1_input", type="number", min= .75 , max= .95 ,  value= .85 , placeholder="theta1", style={'textAlign': 'center', 'fontSize': 18, 'width':  '40%'}),
    dcc.Slider(
        id='theta1_slider',
        min= .75 ,
        max= .95 ,
        step=.01,
        value= .85 ,
        marks={
           .75 : {'label': '.75', 'style': {'color': '#999999', 'fontSize': 15}},
            .95: {'label': '', 'style': {'color': '#999999', 'fontSize': 15}}
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
        html.Label(children = 'note: hover over graph to see Plotly graph controls on top right', style={'textAlign': 'center', 'color': '#b6e0f8', 'fontSize': 17, 'padding': "0px"}),
        html.Label(children = 'Graph 1 Controls:', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 15, 'padding': "0px", 'marginTop' : '30px'}),
        html.Div(id = 'dt1'),
       
        html.Label(children = 'Graph 2 Controls:', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 15, 'padding': "0px", 'marginTop' : '100%'}),
        
        html.Div(id = 'dt2'),
        
        
            ])
    
    ]),
    
    ])
         
    ]),
    
    
    # More info Tab
    dcc.Tab(label='More Info', children=[
        html.Div(style={'textAlign': 'center','display': 'flex', 'flex-direction': 'row'}, children=[
        
            html.Div(style={'padding': 10, 'flex': 1}, children=[
                
                html.Div(children = 'Welcome!', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
                html.Label('Welcome to the COVID-19 web app. The purpose of this app is to allow you to explore various COVID scenarios within a realistic mathematical model for COVID. You can vary a large number of parameters on the \'Controls\' tab which are shown in the table to the right, or just use the defaults for those not of interest to you. For example, you might be interested in the disease itself and variations with those parameters. Or, you are interested in social issues such as mask use and vaccines, and parameters related to those issues.', 
                   style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px"}),
                
                html.Div(children = 'Instructions', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
                html.Div(children = 'Your main control is through the sliders. Reasonable default values for the sliders will appear, and you can change any or all of the values. The sliders are organized roughly in terms of: initial conditions, disease properties, demographics, and interventions.', style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "8px"}),
                html.Div(children = 'Once you change the values of the sliders, simply click on the \'Plots\' tab to see the result. To try a new case, click on the \'Controls\' tab to return to the sliders and try something new.', style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "8px"}),
                html.Div(children = 'The app also allows you to make direct comparisons using two plots for different cases. On the left side of the main page, there is a toggle switch. To make the first plot, leave the toggle where it is and set the sliders as you wish. Go the \'Plots\' tab to see that result. Return the main \'Controls\' tab, switch the toggle to the right, and choose new control values. Go back to the \'Plots\' tab and you will now see two plots, one for each case. This allows you to, for example, compare the impact of mask use or vaccines.', style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "8px"}),
                
                html.Div(children = 'Model flow diagram', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),
                html.Img(src=app.get_asset_url('/newchart.png'), style={'height':'350px', 'width':'660px'}),
                
                html.Div(children = 'Background', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px", 'marginTop' : '10px'}),
                html.Label('The COVID model being used was created by Mingwang Shen, Lei Zhang, and Yan Li with the China-Australia Joint Research Center for Infectious Diseases. To learn more about COVID modeling and this specific model, please read the article cited below.',
                            style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "8px"}),
                html.Label(['This app was developed by Sam Bollman, an undergraduate at Michigan State University (MSU). ', html.A('David Butts', href='https://murillogroupmsu.com/david-j-butts/'), ', a graduate student at MSU, helped with coding and developing the final model we implemented. ', 'Finally, ', html.A('Prof. Murillo', href='https://murillogroupmsu.com/dr-michael-murillo/'), ' lead the project as part of the MSU Professorial Assistant program. If you have any corrections and/or comments, please contact us ', html.A('https://murillogroupmsu.com/dr-michael-murillo/', href='https://murillogroupmsu.com/dr-michael-murillo/'), '.'],
                           style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "8px"}),                
                html.Label(['Shen, Mingwang, et al. “Projected Covid-19 Epidemic in the United States in the Context of the Effectiveness of a Potential Vaccine and Implications for Social Distancing and Face Mask Use.” Vaccine, Elsevier, 27 Feb. 2021, ', html.A('https://www.sciencedirect.com/science/article/pii/S0264410X2100236X?via%3Dihub#!', href='https://www.sciencedirect.com/science/article/pii/S0264410X2100236X?via%3Dihub#!')],
                           style={'textAlign': 'left', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px", 'marginLeft' : '15px'}),
               
                
                
                
                
                
                ]),
                
                html.Div(style={'padding': 10, 'flex': 1}, children=[
                
                html.Div(children = 'Parameters', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px"}),

                           # style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                html.Div( style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "0px"}),
                
                
                            
                dash.dash_table.DataTable(
                    style_data={
                'whiteSpace': 'normal',
                'lineHeight': '15px',
                'textAlign': 'center'
                    },
                    style_header={'backgroundColor': '#bbbbbb','fontWeight': 'bold', 'textAlign': 'center'},
                    style_cell={'backgroundColor': '#444444'},
                    style_cell_conditional=[{'if': {'column_id': 'Description'},'textAlign': 'left'}, {'if': {'column_id': 'Variable'},'textAlign': 'center'}],
                    style_data_conditional=[
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "inherit !important",
                                "border": "inherit !important",
                                }],
                    
                        data=data.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in data.columns]),
                        
                # html.Div(children = 'More information on r0', style={'textAlign': 'center', 'color': '#ffffff', 'fontSize': 20, 'padding': "8px", 'marginTop' : '10px'}),
                # html.Label('r0 or r-naught represents the average amount of people that a single person will infect with a disease. In this model r0 is shown by dividing the infection rate by the recovery rate.',
                #            style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "1px"}),
                # html.Div(children =  'r0 values for common diseases(NPR):' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px", 'marginTop' : '5px'}),
                # html.Div(children =  'influenza: 1-2' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'ebola: 2' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'HIV: 4' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'SARS: 4' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'mumps: 10' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'measles: 18' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'r0 values for COVID-19(BBC): ' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px", 'marginTop' : '10px'}),
                # html.Div(children =  'original virus: 2.4-2.6' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'delta varient: 5-8' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
                # html.Div(children =  'omnicron varient: unknown (likely >8)' ,  style={'textAlign': 'center', 'color': '#bbbbbb', 'fontSize': 15, 'padding': "0px"}),
       
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

@app.callback(
    Output('dt1', 'children'),
    Output('dt2', 'children'),
    Input('N0_slider', 'value'),
    Input('k1_slider', 'value'),
    Input('beta_slider', 'value'),
    Input('ev_slider', 'value'),
    Input('theta1_slider', 'value'),
    Input('m_ini_slider', 'value'),
    Input('r_slider', 'value'),
    Input('gamma0_slider', 'value'),
    Input('compare_toggle', 'value')
    )
def data_table(N0_value, k1_value, beta_value, ev_value, theta1_value, m_ini_value, r_value, gamma0_value, graph_toggle):
    graph2_data = {"Variable": ['population', 'k1', 'beta', 'ev', 'theta1', 'm_ini', 'r', 'gamma0'],
                   "Value": [N0_value, k1_value, beta_value, ev_value, theta1_value, m_ini_value, r_value, gamma0_value]  }
    data2 = pd.DataFrame(graph2_data)
    dt = dash.dash_table.DataTable(
                    style_data={
                'whiteSpace': 'normal',
                'lineHeight': '15px',
                'textAlign': 'center' 
                    },
                    style_header={'backgroundColor': '#bbbbbb','fontWeight': 'bold', 'textAlign': 'center'},
                    style_cell={'backgroundColor': '#444444', 'textAlign': 'center'},
                    style_data_conditional=[
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "inherit !important",
                                "border": "inherit !important",
                                }],
                    
                        data=data2.to_dict('records'),
                        columns=[{'id': d, 'name': d} for d in data2.columns])
    if graph_toggle == False:
        return dt, dash.no_update
    else:
        return dash.no_update,dt
    


# k1 input, slider sync
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
    
    if trigger_id == "range_input":
        range_input = i_value 
    else:
        range_input = s_value
            
    if i_value != None: 
        return range_slider, range_input
    else:
        return s_value, s_value

# k1 input, slider sync
@app.callback(
    Output('k1_slider', 'value'),
    Output('k1_input', 'value'),
    Input('k1_slider', 'value'),
    Input('k1_input', 'value'))
def update_k1_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "k1_slider" :
        k1_slider = s_value
    else:
        k1_slider = i_value
    
    if trigger_id == "k1_input":
        k1_input = i_value 
    else:
        k1_input = s_value
            
    if i_value != None: 
        return k1_slider, k1_input
    else:
        return s_value, s_value
    
# k2 input, slider sync
@app.callback(
    Output('k2_slider', 'value'),
    Output('k2_input', 'value'),
    Input('k2_slider', 'value'),
    Input('k2_input', 'value'))
def update_k2_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "k2_slider" :
        k2_slider = s_value
    else:
        k2_slider = i_value
    
    if trigger_id == "k2_input":
        k2_input = i_value 
    else:
        k2_input = s_value
            
    if i_value != None: 
        return k2_slider, k2_input
    else:
        return s_value, s_value 
    
# r input, slider sync
@app.callback(
    Output('r_slider', 'value'),
    Output('r_input', 'value'),
    Input('r_slider', 'value'),
    Input('r_input', 'value'))
def update_r_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "r_slider" :
        r_slider = s_value
    else:
        r_slider = i_value
    
    if trigger_id == "r_input":
        r_input = i_value 
    else:
        r_input = s_value
            
    if i_value != None: 
        return r_slider, r_input
    else:
        return s_value, s_value 

# gamma1 input, slider sync
@app.callback(
    Output('gamma1_slider', 'value'),
    Output('gamma1_input', 'value'),
    Input('gamma1_slider', 'value'),
    Input('gamma1_input', 'value'))
def update_gamma1_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "gamma1_slider" :
        gamma1_slider = s_value
    else:
        gamma1_slider = i_value
    
    if trigger_id == "gamma1_input":
        gamma1_input = i_value 
    else:
        gamma1_input = s_value
            
    if i_value != None: 
        return gamma1_slider, gamma1_input
    else:
        return s_value, s_value 

# gamm0 input, slider sync
@app.callback(
    Output('gamma0_slider', 'value'),
    Output('gamma0_input', 'value'),
    Input('gamma0_slider', 'value'),
    Input('gamma0_input', 'value'))
def update_gamma0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "gamma0_slider" :
        gamma0_slider = s_value
    else:
        gamma0_slider = i_value
    
    if trigger_id == "gamma0_input":
        gamma0_input = i_value 
    else:
        gamma0_input = s_value
            
    if i_value != None: 
        return gamma0_slider, gamma0_input
    else:
        return s_value, s_value 
# alpha1 input, slider sync
@app.callback(
    Output('alpha1_slider', 'value'),
    Output('alpha1_input', 'value'),
    Input('alpha1_slider', 'value'),
    Input('alpha1_input', 'value'))
def update_alpha1_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "alpha1_slider" :
        alpha1_slider = s_value
    else:
        alpha1_slider = i_value
    
    if trigger_id == "alpha1_input":
        alpha1_input = i_value 
    else:
        alpha1_input = s_value
        
    if i_value != None: 
        return alpha1_slider, alpha1_input
    else:
        return s_value, s_value 

# alpha2 input, slider sync
@app.callback(
    Output('alpha2_slider', 'value'),
    Output('alpha2_input', 'value'),
    Input('alpha2_slider', 'value'),
    Input('alpha2_input', 'value'))
def update_alpha2_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "alpha2_slider" :
        alpha2_slider = s_value
    else:
        alpha2_slider = i_value
    
    if trigger_id == "alpha2_input":
        alpha2_input = i_value 
    else:
        alpha2_input = s_value
        
    if i_value != None: 
        return alpha2_slider, alpha2_input
    else:
        return s_value, s_value 
    
# gamma2 input, slider sync
@app.callback(
    Output('gamma2_slider', 'value'),
    Output('gamma2_input', 'value'),
    Input('gamma2_slider', 'value'),
    Input('gamma2_input', 'value'))
def update_gamma2_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "gamma2_slider" :
        gamma2_slider = s_value
    else:
        gamma2_slider = i_value
    
    if trigger_id == "gamma2_input":
        gamma2_input = i_value 
    else:
        gamma2_input = s_value         
        
    if i_value != None: 
        return gamma2_slider, gamma2_input
    else:
        return s_value, s_value 
    
# xi input, slider sync
@app.callback(
    Output('xi_slider', 'value'),
    Output('xi_input', 'value'),
    Input('xi_slider', 'value'),
    Input('xi_input', 'value'))
def update_xi_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "xi_slider" :
        xi_slider = s_value
    else:
        xi_slider = i_value
    
    if trigger_id == "xi_input":
        xi_input = i_value 
    else:
        xi_input = s_value           
        
    if i_value != None: 
        return xi_slider, xi_input
    else:
        return s_value, s_value 
# V0 input, slider sync
@app.callback(
    Output('V0_slider', 'value'),
    Output('V0_input', 'value'),
    Input('V0_slider', 'value'),
    Input('V0_input', 'value'))
def update_V0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "V0_slider" :
        V0_slider = s_value
    else:
        V0_slider = i_value
    
    if trigger_id == "V0_input":
        V0_input = i_value 
    else:
        V0_input = s_value          
        
    if i_value != None: 
        return V0_slider, V0_input
    else:
        return s_value, s_value 
# Nf0 input, slider sync
@app.callback(
    Output('Nf0_slider', 'value'),
    Output('Nf0_input', 'value'),
    Input('Nf0_slider', 'value'),
    Input('Nf0_input', 'value'))
def update_Nf0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "Nf0_slider" :
        Nf0_slider = s_value
    else:
        Nf0_slider = i_value
    
    if trigger_id == "Nf0_input":
        Nf0_input = i_value 
    else:
        Nf0_input = s_value          
        
    if i_value != None: 
        return Nf0_slider, Nf0_input
    else:
        return s_value, s_value 
# Sf0 input, slider sync
@app.callback(
    Output('Sf0_slider', 'value'),
    Output('Sf0_input', 'value'),
    Input('Sf0_slider', 'value'),
    Input('Sf0_input', 'value'))
def update_Sf0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "Sf0_slider" :
        Sf0_slider = s_value
    else:
        Sf0_slider = i_value
    
    if trigger_id == "Sf0_input":
        Sf0_input = i_value 
    else:
        Sf0_input = s_value          
        
    if i_value != None: 
        return Sf0_slider, Sf0_input
    else:
        return s_value, s_value 
# Vf0 input, slider sync
@app.callback(
    Output('Vf0_slider', 'value'),
    Output('Vf0_input', 'value'),
    Input('Vf0_slider', 'value'),
    Input('Vf0_input', 'value'))
def update_Vf0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "Vf0_slider" :
        Vf0_slider = s_value
    else:
        Vf0_slider = i_value
    
    if trigger_id == "Vf0_input":
        Vf0_input = i_value 
    else:
        Vf0_input = s_value          
        
    if i_value != None: 
        return Vf0_slider, Vf0_input
    else:
        return s_value, s_value 
# T10 input, slider sync
@app.callback(
    Output('T10_slider', 'value'),
    Output('T10_input', 'value'),
    Input('T10_slider', 'value'),
    Input('T10_input', 'value'))
def update_T10_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "T10_slider" :
        T10_slider = s_value
    else:
        T10_slider = i_value
    
    if trigger_id == "T10_input":
        T10_input = i_value 
    else:
        T10_input = s_value          
        
    if i_value != None: 
        return T10_slider, T10_input
    else:
        return s_value, s_value 
# T20 input, slider sync
@app.callback(
    Output('T20_slider', 'value'),
    Output('T20_input', 'value'),
    Input('T20_slider', 'value'),
    Input('T20_input', 'value'))
def update_T20_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "T20_slider" :
        T20_slider = s_value
    else:
        T20_slider = i_value
    
    if trigger_id == "T20_input":
        T20_input = i_value 
    else:
        T20_input = s_value          
        
    if i_value != None: 
        return T20_slider, T20_input
    else:
        return s_value, s_value 
# D0 input, slider sync
@app.callback(
    Output('D0_slider', 'value'),
    Output('D0_input', 'value'),
    Input('D0_slider', 'value'),
    Input('D0_input', 'value'))
def update_D0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "D0_slider" :
        D0_slider = s_value
    else:
        D0_slider = i_value
    
    if trigger_id == "D0_input":
        D0_input = i_value 
    else:
        D0_input = s_value          
        
    if i_value != None: 
        return D0_slider, D0_input
    else:
        return s_value, s_value 
# R0 input, slider sync
@app.callback(
    Output('R0_slider', 'value'),
    Output('R0_input', 'value'),
    Input('R0_slider', 'value'),
    Input('R0_input', 'value'))
def update_R0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "R0_slider" :
        R0_slider = s_value
    else:
        R0_slider = i_value
    
    if trigger_id == "R0_input":
        R0_input = i_value 
    else:
        R0_input = s_value          
        
    if i_value != None: 
        return R0_slider, R0_input
    else:
        return s_value, s_value 
# C0 input, slider sync
@app.callback(
    Output('C0_slider', 'value'),
    Output('C0_input', 'value'),
    Input('C0_slider', 'value'),
    Input('C0_input', 'value'))
def update_C0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "C0_slider" :
        C0_slider = s_value
    else:
        C0_slider = i_value
    
    if trigger_id == "C0_input":
        C0_input = i_value 
    else:
        C0_input = s_value          
        
    if i_value != None: 
        return C0_slider, C0_input
    else:
        return s_value, s_value
# N0 input, slider sync
@app.callback(
    Output('N0_slider', 'value'),
    Output('N0_input', 'value'),
    Input('N0_slider', 'value'),
    Input('N0_input', 'value'))
def update_N0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "N0_slider" :
        N0_slider = s_value
    else:
        N0_slider = i_value
    
    if trigger_id == "N0_input":
        N0_input = i_value 
    else:
        N0_input = s_value          
        
    if i_value != None: 
        return N0_slider, N0_input
    else:
        return s_value, s_value
# E0 input, slider sync
@app.callback(
    Output('E0_slider', 'value'),
    Output('E0_input', 'value'),
    Input('E0_slider', 'value'),
    Input('E0_input', 'value'))
def update_E0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "E0_slider" :
        E0_slider = s_value
    else:
        E0_slider = i_value
    
    if trigger_id == "E0_input":
        E0_input = i_value 
    else:
        E0_input = s_value          
        
    if i_value != None: 
        return E0_slider, E0_input
    else:
        return s_value, s_value 
    
# A0 input, slider sync
@app.callback(
    Output('A0_slider', 'value'),
    Output('A0_input', 'value'),
    Input('A0_slider', 'value'),
    Input('A0_input', 'value'))
def update_A0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
      
    if trigger_id == "A0_slider" :
        A0_slider = s_value
    else:
        A0_slider = i_value
    
    if trigger_id == "A0_input":
        A0_input = i_value 
    else:
        A0_input = s_value
                   
    if i_value != None: 
        return A0_slider, A0_input
    else:
        return s_value, s_value 
    
# I10 input, slider sync
@app.callback(
    Output('I10_slider', 'value'),
    Output('I10_input', 'value'),
    Input('I10_slider', 'value'),
    Input('I10_input', 'value'))
def update_I10_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "I10_slider" :
        I10_slider = s_value
    else:
        I10_slider = i_value
    
    if trigger_id == "I10_input":
        I10_input = i_value 
    else:
        I10_input = s_value           
        
    if i_value != None: 
        return I10_slider, I10_input
    else:
        return s_value, s_value 
    
# I20 input, slider sync
@app.callback(
    Output('I20_slider', 'value'),
    Output('I20_input', 'value'),
    Input('I20_slider', 'value'),
    Input('I20_input', 'value'))
def update_I20_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "I20_slider" :
        I20_slider = s_value
    else:
        I20_slider = i_value
    
    if trigger_id == "I20_input":
        I20_input = i_value 
    else:
        I20_input = s_value           
        
    if i_value != None: 
        return I20_slider, I20_input
    else:
        return s_value, s_value 
    
# beta input, slider sync
@app.callback(
    Output('beta_slider', 'value'),
    Output('beta_input', 'value'),
    Input('beta_slider', 'value'),
    Input('beta_input', 'value'))
def update_beta_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "beta_slider" :
        beta_slider = s_value
    else:
        beta_slider = i_value
    
    if trigger_id == "beta_input":
        beta_input = i_value 
    else:
        beta_input = s_value            
        
    if i_value != None: 
        return beta_slider, beta_input
    else:
        return s_value, s_value 
    
# epsilon input, slider sync
@app.callback(
    Output('epsilon_slider', 'value'),
    Output('epsilon_input', 'value'),
    Input('epsilon_slider', 'value'),
    Input('epsilon_input', 'value'))
def update_epsilon_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "epsilon_slider" :
        epsilon_slider = s_value
    else:
        epsilon_slider = i_value
    
    if trigger_id == "epsilon_input":
        epsilon_input = i_value 
    else:
        epsilon_input = s_value            
        
    if i_value != None: 
        return epsilon_slider, epsilon_input
    else:
        return s_value, s_value 
    
# rho input, slider sync
@app.callback(
    Output('rho_slider', 'value'),
    Output('rho_input', 'value'),
    Input('rho_slider', 'value'),
    Input('rho_input', 'value'))
def update_rho_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
       
    if trigger_id == "rho_slider" :
        rho_slider = s_value
    else:
        rho_slider = i_value
    
    if trigger_id == "rho_input":
        rho_input = i_value 
    else:
        rho_input = s_value           
        
    if i_value != None: 
        return rho_slider, rho_input
    else:
        return s_value, s_value 
    
# k3 input, slider sync
@app.callback(
    Output('k3_slider', 'value'),
    Output('k3_input', 'value'),
    Input('k3_slider', 'value'),
    Input('k3_input', 'value'))
def update_k3_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "k3_slider" :
        k3_slider = s_value
    else:
        k3_slider = i_value
    
    if trigger_id == "k3_input":
        k3_input = i_value 
    else:
        k3_input = s_value          
        
    if i_value != None: 
        return k3_slider, k3_input
    else:
        return s_value, s_value 
    
# m_ini input, slider sync
@app.callback(
    Output('m_ini_slider', 'value'),
    Output('m_ini_input', 'value'),
    Input('m_ini_slider', 'value'),
    Input('m_ini_input', 'value'))
def update_m_ini_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "m_ini_slider" :
        m_ini_slider = s_value
    else:
        m_ini_slider = i_value
    
    if trigger_id == "m_ini_input":
        m_ini_input = i_value 
    else:
        m_ini_input = s_value            
        
    if i_value != None: 
        return m_ini_slider, m_ini_input
    else:
        return s_value, s_value 
    
# m0 input, slider sync
@app.callback(
    Output('m0_slider', 'value'),
    Output('m0_input', 'value'),
    Input('m0_slider', 'value'),
    Input('m0_input', 'value'))
def update_m0_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "m0_slider" :
        m0_slider = s_value
    else:
        m0_slider = i_value
    
    if trigger_id == "m0_input":
        m0_input = i_value 
    else:
        m0_input = s_value            
        
    if i_value != None: 
        return m0_slider, m0_input
    else:
        return s_value, s_value 
    
# t_ini input, slider sync
@app.callback(
    Output('t_ini_slider', 'value'),
    Output('t_ini_input', 'value'),
    Input('t_ini_slider', 'value'),
    Input('t_ini_input', 'value'))
def update_t_ini_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if trigger_id == "t_ini_slider" :
        t_ini_slider = s_value
    else:
        t_ini_slider = i_value
    
    if trigger_id == "t_ini_input":
        t_ini_input = i_value 
    else:
        t_ini_input = s_value
        
    if i_value != None: 
        return t_ini_slider, t_ini_input
    else:
        return s_value, s_value 
    
# t_reopen input, slider sync
@app.callback(
    Output('t_reopen_slider', 'value'),
    Output('t_reopen_input', 'value'),
    Input('t_reopen_slider', 'value'),
    Input('t_reopen_input', 'value'))
def update_t_reopen_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "t_reopen_slider" :
        t_reopen_slider = s_value
    else:
        t_reopen_slider = i_value
    
    if trigger_id == "t_reopen_input":
        t_reopen_input = i_value 
    else:
        t_reopen_input = s_value
            
    if i_value != None: 
        return t_reopen_slider, t_reopen_input
    else:
        return s_value, s_value 
    
# q_ini input, slider sync
@app.callback(
    Output('q_ini_slider', 'value'),
    Output('q_ini_input', 'value'),
    Input('q_ini_slider', 'value'),
    Input('q_ini_input', 'value'))
def update_q_ini_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "q_ini_slider" :
        q_ini_slider = s_value
    else:
        q_ini_slider = i_value
    
    if trigger_id == "q_ini_input":
        q_ini_input = i_value 
    else:
        q_ini_input = s_value
            
    if i_value != None: 
        return q_ini_slider, q_ini_input
    else:
        return s_value, s_value 
    
# qbar input, slider sync
@app.callback(
    Output('qbar_slider', 'value'),
    Output('qbar_input', 'value'),
    Input('qbar_slider', 'value'),
    Input('qbar_input', 'value'))
def update_qbar_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "qbar_slider" :
        qbar_slider = s_value
    else:
        qbar_slider = i_value
    
    if trigger_id == "qbar_input":
        qbar_input = i_value 
    else:
        qbar_input = s_value
            
    if i_value != None: 
        return qbar_slider, qbar_input
    else:
        return s_value, s_value 
    
# theta2 input, slider sync
@app.callback(
    Output('theta2_slider', 'value'),
    Output('theta2_input', 'value'),
    Input('theta2_slider', 'value'),
    Input('theta2_input', 'value'))
def update_theta2_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "theta2_slider" :
        theta2_slider = s_value
    else:
        theta2_slider = i_value
    
    if trigger_id == "theta2_input":
        theta2_input = i_value 
    else:
        theta2_input = s_value
        
    if i_value != None: 
        return theta2_slider, theta2_input
    else:
        return s_value, s_value 
    
# p_ini input, slider sync
@app.callback(
    Output('p_ini_slider', 'value'),
    Output('p_ini_input', 'value'),
    Input('p_ini_slider', 'value'),
    Input('p_ini_input', 'value'))
def update_p_ini_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "p_ini_slider" :
        p_ini_slider = s_value
    else:
        p_ini_slider = i_value
    
    if trigger_id == "p_ini_input":
        p_ini_input = i_value 
    else:
        p_ini_input = s_value
            
    if i_value != None: 
        return p_ini_slider, p_ini_input
    else:
        return s_value, s_value 
    
# pbar input, slider sync
@app.callback(
    Output('pbar_slider', 'value'),
    Output('pbar_input', 'value'),
    Input('pbar_slider', 'value'),
    Input('pbar_input', 'value'))
def update_pbar_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "pbar_slider" :
        pbar_slider = s_value
    else:
        pbar_slider = i_value
    
    if trigger_id == "pbar_input":
        pbar_input = i_value 
    else:
        pbar_input = s_value
        
    if i_value != None: 
        return pbar_slider, pbar_input
    else:
        return s_value, s_value 
    
# tp input, slider sync
@app.callback(
    Output('tp_slider', 'value'),
    Output('tp_input', 'value'),
    Input('tp_slider', 'value'),
    Input('tp_input', 'value'))
def update_tp_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "tp_slider" :
        tp_slider = s_value
    else:
        tp_slider = i_value
    
    if trigger_id == "tp_input":
        tp_input = i_value 
    else:
        tp_input = s_value
        
    if i_value != None: 
        return tp_slider, tp_input
    else:
        return s_value, s_value 
    
# p2 input, slider sync
@app.callback(
    Output('p2_slider', 'value'),
    Output('p2_input', 'value'),
    Input('p2_slider', 'value'),
    Input('p2_input', 'value'))
def update_p2_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "p2_slider" :
        p2_slider = s_value
    else:
        p2_slider = i_value
    
    if trigger_id == "p2_input":
        p2_input = i_value 
    else:
        p2_input = s_value
        
    if i_value != None: 
        return p2_slider, p2_input
    else:
        return s_value, s_value 
    
# theta1 input, slider sync
@app.callback(
    Output('theta1_slider', 'value'),
    Output('theta1_input', 'value'),
    Input('theta1_slider', 'value'),
    Input('theta1_input', 'value'))
def update_theta1_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "theta1_slider" :
        theta1_slider = s_value
    else:
        theta1_slider = i_value
    
    if trigger_id == "theta1_input":
        theta1_input = i_value 
    else:
        theta1_input = s_value
            
    if i_value != None: 
        return theta1_slider, theta1_input
    else:
        return s_value, s_value 
    
# mu1 input, slider sync
@app.callback(
    Output('mu1_slider', 'value'),
    Output('mu1_input', 'value'),
    Input('mu1_slider', 'value'),
    Input('mu1_input', 'value'))
def update_mu1_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "mu1_slider" :
        mu1_slider = s_value
    else:
        mu1_slider = i_value
    
    if trigger_id == "mu1_input":
        mu1_input = i_value 
    else:
        mu1_input = s_value
            
    if i_value != None: 
        return mu1_slider, mu1_input
    else:
        return s_value, s_value 
    
# mu2 input, slider sync
@app.callback(
    Output('mu2_slider', 'value'),
    Output('mu2_input', 'value'),
    Input('mu2_slider', 'value'),
    Input('mu2_input', 'value'))
def update_mu2_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "mu2_slider" :
        mu2_slider = s_value
    else:
        mu2_slider = i_value
    
    if trigger_id == "mu2_input":
        mu2_input = i_value 
    else:
        mu2_input = s_value
        
    if i_value != None: 
        return mu2_slider, mu2_input
    else:
        return s_value, s_value 
    
# w input, slider sync
@app.callback(
    Output('w_slider', 'value'),
    Output('w_input', 'value'),
    Input('w_slider', 'value'),
    Input('w_input', 'value'))
def update_w_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "w_slider" :
        w_slider = s_value
    else:
        w_slider = i_value
    
    if trigger_id == "w_input":
        w_input = i_value 
    else:
        w_input = s_value
                  
    if i_value != None: 
        return w_slider, w_input
    else:
        return s_value, s_value 
    
# ev input, slider sync
@app.callback(
    Output('ev_slider', 'value'),
    Output('ev_input', 'value'),
    Input('ev_slider', 'value'),
    Input('ev_input', 'value'))
def update_ev_output(s_value, i_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "ev_slider" :
        ev_slider = s_value
    else:
        ev_slider = i_value
    
    if trigger_id == "ev_input":
        ev_input = i_value 
    else:
        ev_input = s_value
            
    if i_value != None: 
        return ev_slider, ev_input
    else:
        return s_value, s_value 
    


# Compare toggle
@app.callback(
    Output('compare_toggle_output', 'children'),
    Input('compare_toggle', 'value')
)
def update_output(value):
    if value == False:
        value = "top graph"
    else:
        value = "bottom graph"
    return 'you are editing the {}'.format(value) 


# Update Graph callback, interaction
@app.callback(
    Output('SIR-graph', 'figure'),
    Output('SIR-graph2', 'figure'),
    
    Input('k1_slider', 'value'),
    Input('k2_slider', 'value'),
    Input('r_slider', 'value'),
    
    Input('gamma1_slider', 'value'),
    Input('gamma0_slider', 'value'),
    Input('alpha1_slider', 'value'),
    Input('alpha2_slider', 'value'),
    Input('gamma2_slider', 'value'),
    
    Input('xi_slider', 'value'),
    Input('E0_slider', 'value'),
    Input('A0_slider', 'value'),
    Input('I10_slider', 'value'),
    Input('I20_slider', 'value'),
    
    Input('beta_slider', 'value'),
    Input('epsilon_slider', 'value'),
    Input('rho_slider', 'value'),
    Input('k3_slider', 'value'),
    Input('m_ini_slider', 'value'),
    
    Input('m0_slider', 'value'),
    Input('t_ini_slider', 'value'),
    Input('t_reopen_slider', 'value'),
    Input('q_ini_slider', 'value'),
    Input('qbar_slider', 'value'),
    
    Input('theta2_slider', 'value'),
    Input('p_ini_slider', 'value'),
    Input('pbar_slider', 'value'),
    Input('tp_slider', 'value'),
    Input('p2_slider', 'value'),
    
    Input('theta1_slider', 'value'),
    Input('mu1_slider', 'value'),
    Input('mu2_slider', 'value'),
    Input('w_slider', 'value'),
    Input('ev_slider', 'value'),
    
    Input('V0_slider', 'value'),
    Input('Nf0_slider', 'value'),
    Input('Sf0_slider', 'value'),
    Input('Vf0_slider', 'value'),
    Input('T10_slider', 'value'),
    Input('T20_slider', 'value'),
    Input('D0_slider', 'value'),
    Input('R0_slider', 'value'),
    Input('C0_slider', 'value'),
    Input('N0_slider', 'value'),


    Input('compare_toggle', 'value') ,
    Input('range_slider', 'value') 

    )

def update_graph(k1_value, k2_value,  r_value, gamma1_value, gamma0_value, alpha1_value, alpha2_value, gamma2_value,
                 xi_value, E0_value, A0_value, I10_value, I20_value, beta_value, epsilon_value, rho_value, k3_value, 
                 m_ini_value, m0_value, t_ini_value, t_reopen_value,  q_ini_value, qbar_value, theta2_value, 
                 p_ini_value, pbar_value, tp_value, p2_value, theta1_value,  mu1_value, mu2_value, w_value, ev_value, 
                 V0_value, Nf0_value, Sf0_value, Vf0_value, T10_value, T20_value, D0_value, R0_value, C0_value, N0_value, 
                 graph_toggle, range_value):
    
    k1 = 1/k1_value         # The mean incubation time (days)
    k2 = 1/k2_value                  # The mean time from mild/moderate stage to severe/critical stage (days)
    r = r_value                       # The mean number of members in a family
    gamma1 = 1/gamma1_value                # The average recovery period for diagnosed mild/moderate cases (days)
    gamma0 = 1/gamma0_value           # The mean time for natural recovery (days)
    alpha1 = 1/alpha1_value           # The average period from symptoms onset to diagnose for mild/moderate cases (days)
    alpha2 = 1/alpha2_value           # The average diagnose period for severe/critical cases (days)
    gamma2 = 1/(gamma2_value - 1/alpha2)  # The average recovery period for diagnosed severe/critical cases
    xi = 1/xi_value                   # The mean recovery periodfor infected family members (days)
    N0 = N0_value             # 
    
    V0 = V0_value                      # 
    Nf0 = Nf0_value                    # 
    Sf0 = Sf0_value                    # 
    Vf0 = Vf0_value                    # 
    E0 = E0_value                # The initial value of latent individuals
    A0 = A0_value                    # The initial value of asymptomatic individuals
    I10 = I10_value                    # The initial value of undiagnosed mild/moderate cases
    I20 =  I20_value                  # The initial value of undiagnosed severe/critical individuals
    T10 = T10_value                  # 
    T20 = T20_value                     # 
    R0 = R0_value                      # 
    D0 = D0_value                     # 
    C0 = C0_value                        # confirmed cases
    S0 = N0 - V0 - E0 - A0 - I10 - I20 - T10 - T20 - R0
    
    beta = beta_value                # The per-act transmission probability in contact withinfected individuals with symptoms
    epsilon = epsilon_value               # The reduction in per-act transmission probability ifinfection is in latent and asymptomatic stage
    rho = rho_value                 # The probability that an individual is asymptomatic
    k3 = k3_value*k2               # The progression rate from diagnosed mild/moderate stage to diagnosed severe/critical stage
    
    m_ini = m_ini_value                # Base daily contact number in the public settings
    m0 = m0_value                  # Change rate of daily contact number
    t_ini = t_ini_value             # The time when the contact number is half of maximal and minimal contact number in public settings  (before reopening)
    t_reopen = t_reopen_value              # The time when the contact number is half of maximal and minimal contact number in public settings (after reopening)
     
    q_ini = q_ini_value                 # Base percentage of handwashing before the epidemic
    qbar = qbar_value                 # Maximal percentage of handwashing during the epidemic
     
    theta2 = theta2_value                # The effectiveness of handwashing in preventing infection
    p_ini = p_ini_value                   # Base percentage of face mask usage in the public settings before the Executive Order on face mask use
    pbar = pbar_value                 # Percentage of face mask usage in the public settings after the Executive Order on face mask use
    tp = tp_value                     # The time when face mask usage in the public settings is half of the maximal face mask usage rate
    p2 = p2_value                      # The usage percentage of mask in the households
    theta1 = theta1_value                # The effectiveness of mask in preventing infection
    mu1 = mu1_value                   # Disease-induced death rate of undiagnosed severe/critical cases
    mu2 = mu2_value*mu1             # Disease-induced death rate of diagnosed severe/critical cases
     
    w = w_value               # vaccination rate 
    ev = ev_value              # vaccine effectivness 
    time_range = [0, range_value]

    
    res = solve_ivp(covid_model, time_range, [S0, V0, Nf0, Sf0, Vf0, E0, A0, I10, I20, T10, T20, R0, D0, C0], args=[k1,k2,r,gamma1,gamma0,alpha1,alpha2,gamma2,xi,beta,epsilon,rho,k3,m_ini,m0,t_ini,t_reopen,q_ini,qbar,theta2,p_ini,pbar,tp,p2,theta1,mu1,mu2,w, ev] ,dense_output=True)


    
    tt = np.linspace(0,range_value,600) 
    
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    
    ax.plot(tt,res.sol(tt)[0],label='susceptible', color='#005d8f')
    ax.plot(tt,res.sol(tt)[1],label='vaccinated') 
    ax.plot(tt,res.sol(tt)[5],label='latent infections') 
    ax.plot(tt,res.sol(tt)[6],label='asymptomatic infections') 
    ax.plot(tt,res.sol(tt)[7]+res.sol(tt)[8],label='undiagnosed infections') 
    ax.plot(tt,res.sol(tt)[9]+res.sol(tt)[10],label='diagnosed infections') 
    ax.plot(tt,res.sol(tt)[11],label='recovered', color='#8f71eb') 
    ax.plot(tt,res.sol(tt)[12],label='dead', color='#de4e4e') 
    
    
    ax.legend()
    plt.xlabel("time")
    ax.grid(True)
    
    plotly_fig = mpl_to_plotly(fig)

    
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


if __name__ == '__main__':
    app.run_server(debug = True)
    
    
    
    
