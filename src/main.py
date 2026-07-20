import streamlit as st
from sympy.parsing.latex import parse_latex
import sympy
from sympy import Derivative, Symbol, Function, Mul, symbols
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from elara_symbolic.cas import *
from st_mathlive import mathfield
import polars as pl
from PIL import Image
from differential_equation import *
import numpy as np
import re

if "app_loaded" not in st.session_state:
    st.toast("App loading...", icon="ℹ", duration=2)
    st.session_state["app_loaded"] = True

# takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: set, Y0: str, solver: str):
    Y0 = [float(i) for i in Y0.split(",")]
    if upperRange <= lowerRange:
        st.write("Unable to display equation: lower bound is greater than or equal to upper bound.")
    else:
        # Warn user; since it does actually take a while to solve
        # and we don't want the user to think nothing is happening
        st.toast("Solving might be slow and take a while",
	                 icon="ℹ",
                 duration="short")
        with st.spinner("Solving in progress...", show_time=True) as status:
            solve_complete = False
            de_sols = None
            # Crude way of blocking execution of further code
            # while the differential equation is solved
            while not solve_complete:
                try:
                    diffeq = Differential_Equation(constantValues, Tex)
                    de_sols = Differential_Equation_Solution(diffeq, upperRange, lowerRange, stepSize, Y0, solver)
                except ValueError as e:
                    st.error(f"Solve unsuccessful: {str(e)}")
                solve_complete = True
            if de_sols:
                st.session_state["valid_equation"] = True
                #create a dataframe containing the date and time and plot that dataframe
                # this dataframe can be pretty slow to
                # initialize though, so this
                # is used here to prevent the UI elements
                # from being displayed until the dataframe
                # is successfully populated
                st.session_state["ode_solution"] = de_sols.plotDF
                st.session_state["functions"] = de_sols.functions
                st.session_state["latex"] = diffeq.latex
                print(de_sols.plotDF.head(5))
            else:
                #this converts our sympy back into latex so it can be displayed again to the human eye so
                #accuracy can be confirmed
                st.session_state["valid_equation"] = False

if "ode_solution" in st.session_state:
    if st.session_state["valid_equation"]:
        st.success("Solve successful! Plotting solution...")
        st.write(rf"**Numerical solution to** ${st.session_state["latex"]}$")
        if len(st.session_state['functions']) > 1:
            selected_funcs = st.multiselect("Select Derivatives You Want to Use", st.session_state["functions"], default=st.session_state["functions"][0])
        else:
            selected_funcs = st.session_state["functions"]
        st.line_chart(data=st.session_state["ode_solution"], x='x', y=selected_funcs)
    else:
        st.write("Invalid differential equation: ")
        st.latex(st.session_state["Tex"])
        st.write("please enter a valid differential equation")
    if st.button("Solve another differential equation"):
        st.session_state.clear()
        st.rerun()
    else:
        st.stop()
else:
    st.write("""
    # Infinitum

    An interactive differential equations solver, developed by [Project Elara](https://elaraproject.org/). Enter a differential equation and Infinitum will numerically solve it for you! Source code is available on our [Codeberg repository](https://codeberg.org/elaraproject/elara-symbolic-ui/)

    :warning: Be aware that the app currently only supports first-order ODEs with $y(x)$ as the dependent variable, and it currently [does not work on Firefox](https://codeberg.org/elaraproject/elara-symbolic-ui/issues/31). Also, the app is _highly experimental_, so if you encounter bugs please [report them to us](https://codeberg.org/elaraproject/elara-symbolic-ui/issues)!
    """)

    # Default ODE is the logistic equation
    default_ode = r"\frac{dy}{dx} = y(1 - y)"
    Tex = default_ode

    equation_to_load = st.session_state["diffeq"] if "diffeq" \
                        in st.session_state else default_ode

    # Code for preliminary processing of the LaTeX
    Tex, _ = mathfield(title="Enter Equations Here",
            value=equation_to_load, 
            mathml_preview=True, upright=False)
    # Pause execution if equation is not yet parsed
    if not Tex:
        st.stop()
    
    # code for selecting what will be a constant and setting the value of said constant
    selected_constants = st.multiselect(label="Enter List of Constants", options=list('abcdefghijklmnopqrstuvwxyz'))
    constant_values = {i : 0.0 for i in selected_constants}
    if type(constant_values) == None: constant_values = {}
    for letter in selected_constants:
        constant_values[letter] = st.number_input(label=f"enter constant value for {letter}: ", value=0.0)

    # This is the code for the components letting the user set the bounds of the graph
    lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
    upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)
    stepSize = st.number_input(label="Enter Step Interval: ", value=0.01)
    Y0 = st.text_input(label="Enter Y0 of the Differential Equation as a Comma Separated List (e.g. 0,1,2,3...): ", value="0.5") # Set the initial condition
    selected_constants = st.selectbox(label="Enter The Solving Method You Wish To Use Here:", options=["Base", "Leapfrog", "RK4"])

    st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0, selected_constants))