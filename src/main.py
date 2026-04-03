import streamlit as st
from sympy.parsing.latex import parse_latex
from sympy import Derivative
from elara_symbolic.calculate import *
from st_mathlive import mathfield
from pandas import DataFrame as df

#function for processing the mathlive user input into text that can be process by sympy
def process_raw_text(Tex: str):
    #we need this here because the mathfield often processes a simple d as this differentialD
    #which confuses sympy so we replace that with the simple d.
    newTex = Tex.replace(r"\differentialD", "d")
    #parse for upright equations TODO: Remove the option for upright equations
    if (r"\mathrm{" in newTex):
        newTex = newTex[8:-1]

    return newTex

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float):
    #parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    #create a dictionary of constants for the parsing
    parseConstants = { i : Symbol(i, constant=True, real=True) for i in constantValues.keys() }

    #substitute whatever y symbol that sympy parses with a 
    functions = [
        Function(derivative.args[0])(derivative.args[1][0])
        for derivative in expr.atoms(Derivative)
    ]
    if len(functions) != 1:
        raise ValueError("ERROR READING: DIFFERENTIAL EQUATION SHOULD HAVE ONE AND ONLY ONE FUNCTION\n")
    for i in functions:
        expr = expr.subs(Symbol(i.func.__name__), i)
    for i in range(len(expr.args)):
        if len(expr.args[i].args) == 1 and expr.args[i].func != functions[0]:
            e = Symbol(str(expr.args[i].func)) * expr.args[i].args[0]
            expr = expr.xreplace({expr.args[i]: e})

    # define a dummy constant for solving the differential equation
    k = Symbol("k", constant=True, real=True)
    #solve the differential equation itself
    constantPass = [(parseConstants[i], constant_values[i]) for i in constant_values.keys()] if len(constant_values) > 0 else [(k, 1.0)]
    de_sols = solve_ode(expr, functions[0], y0=Y0, t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)

    return de_sols

#takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float):
    if upperRange <= lowerRange:
        st.write("Unable to display equation: lower bound is greater than or equal to lower bound ")
    else:
        de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex, constantValues, Y0)
        if type(de_sols) != type(None):
            #create a dataframe containing the date and time and plot that dataframe
            plotDF = df(de_sols['y'], de_sols['t']) 
            st.line_chart(data=plotDF)
        else:
            #this converts our sympy back into latex so it can be displayed again to the human eye so
            #accuracy can be confirmed
            st.write("Invalid differential equation: ")
            st.latex(Tex)
            st.write("please enter a valid differential equation")

st.write("""
# Elara-symbolic UI

Currently being developed...
""")
#Code for preliminary processing of the LaTeX
Tex, _ = mathfield(title="Enter Equations Here", value=r"\frac{dy}{dx} = y(1 - y)", mathml_preview=True, upright=False)

#code for selecting what will be a constant and setting the value of said constant
selected_constants = st.multiselect(label="Enter List of Constants", options=list('abcdefghijklmnopqrstuvwxyz'))
constant_values = {i : 0 for i in selected_constants}
for letter in selected_constants:
    constant_values[letter] = st.number_input(label=f"enter constant value for {letter}: ", value=0.0)
#This is the code for the components letting the user set the bounds of the graph
lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)
stepSize = st.number_input(label="Enter Step Interval: ", value=0.01)
Y0 = st.number_input(label="Enter Y0 of the Differential Equation: ", value=0.5) #Set the initial condition

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0))