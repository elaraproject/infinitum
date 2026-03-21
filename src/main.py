import streamlit as st
from sympy.parsing.latex import parse_latex
from sympy import Derivative
from elara_symbolic.calculate import *
from st_mathlive import mathfield

#function for processing the mathlive user input into text that can be process by sympy
def process_raw_text(Tex: str):
    #we need this here because the mathfield often processes a simple d as this differentialD
    #which confuses sympy so we replace that with the simple d.
    newTex = Tex.replace(r"\differentialD", "d")
    #parse for upright equations TODO: Remove the option for upright equations
    if (r"\mathrm{" in newTex):
        newTex = newTex[8:-1]

    print(newTex)
    return newTex

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: int, Tex: str):
    #parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    #substitute whatever y symbol that sympy parses with a 
    functions = [
        Function(derivative.args[0])(derivative.args[1][0])
        for derivative in expr.atoms(Derivative)
    ]
    for i in functions:
        expr = expr.subs(Symbol(i.func.__name__), i)

    # define a dummy constant for solving the differential equation
    k = Symbol("k", constant=True, real=True)
    #solve the differential equation itself
    print(expr)
    de_sols = solve_ode(expr, functions[0], y0=0.0, t_span=(lowerRange, upperRange), constants=[(k, 1.0)], step_size=stepSize)

    return de_sols

#takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str):
    if upperRange <= lowerRange:
        st.write("Unable to display equation: lower bound is greater than or equal to lower bound ")
    else:
        de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex)
        if type(de_sols) != type(None):
            st.line_chart(de_sols['y'])
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

Tex, _ = mathfield(title="Enter Equations Here", value=r"\frac{dy}{dx} = y(1 - y)", mathml_preview=True, upright=False)

#Code for preliminary processing of the LaTeX
Tex = process_raw_text(Tex)

#This is the code for the components letting the user set the bounds of the graph
lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)
stepSize = st.number_input(label="Enter Step Interval: ", value=0.01)

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex))