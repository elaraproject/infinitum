import streamlit as st
from bs4 import BeautifulSoup
from sympy.parsing.latex import parse_latex
from elara_symbolic.calculate import *
from st_mathlive import mathfield

#function for processing the mathlive user input into text that can be process by sympy
def process_raw_text(Tex: str):
    #NOTE: BUG EXISTS WHERE IT MAY HAVE TROUBLE PARSING CERTAIN UPRIGHT EQUATIONS, DO NOT CHECK THAT BOX
    newTex = Tex.replace(r"\differentialD", "d")
    #parse for upright equations TODO: Remove the option for upright equations
    if (newTex[0:8] == "\mathrm{"):
        newTex = newTex[8:-1]

    return newTex

#solves the differential equation provided as raw LaTeX text
def solve_differential_equation(Tex: str):
    #parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    # define your unknown function
    # to be solved for
    # here it is y(x)
    y = Function("y")(x)
    # define a dummy constant for solving the differential equation
    k = Symbol("k", constant=True, real=True)
    #solve the differential equation itself
    de_sols = solve_ode(expr, y, y0=0.0, t_span=(lowerRange, upperRange), constants=[(k, 1.0)], step_size=0.01)

    return de_sols

#takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, Tex: str):
    if upperRange <= lowerRange:
        st.write("Unable to display equation: lower bound is greater than or equal to lower bound ")
    else:
        de_sols = solve_differential_equation(Tex)
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

Tex, MathML = mathfield(title="Enter Equations Here", value=r"\frac{dy}{dx} = y(1 - y)", mathml_preview=True, upright=False)

#Code for preliminary processing of the LaTeX
Tex = process_raw_text(Tex)

#This is the code for the components letting the user set the bounds of the graph
lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, Tex))