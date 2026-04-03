import streamlit as st
from sympy.parsing.latex import parse_latex, LaTeXParsingError
from sympy import Derivative, Function, Symbol
from elara_symbolic.cas import solve_ode
from st_mathlive import mathfield
from pandas import DataFrame as df

st.toast("App loading...", icon="ℹ️", duration="short")

#function for processing the mathlive user input into text that can be process by sympy
def process_raw_text(Tex: str):
    #we need this here because the mathfield often processes a simple d as this differentialD
    #which confuses sympy so we replace that with the simple d.
    newTex = Tex.replace(r"\differentialD", "d")
    #parse for upright equations TODO: Remove the option for upright equations
    if (r"\mathrm{" in newTex):
        newTex = newTex[8:-1]

    return newTex

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: int, Tex: str):
    #parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
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
    de_sols = solve_ode(expr, functions[0], y0=0.5, t_span=(lowerRange, upperRange), constants=[(k, 1.0)], step_size=stepSize)

    return de_sols

#takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str):
    if upperRange <= lowerRange:
        st.write("Unable to display equation: lower bound is greater than or equal to lower bound ")
    else:
        # Warn user; since it does actually take a while to solve
        # and we don't want the user to think nothing is happening
        st.toast("Solving might be slow and take a while",
                 icon="ℹ️",
                 duration="short")
        with st.spinner("Solving in progress...", show_time=True) as status:
            solve_complete = False
            de_sols = None
            # Crude way of blocking execution of further code
            # while the differential equation is solved
            while not solve_complete:
                try:
                    de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex)
                except ValueError as e:
                    st.error(f"Solve unsuccessful: {str(e)}")
                solve_complete = True
            if de_sols:
                #create a dataframe containing the date and time and plot that dataframe
                # this dataframe can be pretty slow to
                # initialize though, so this
                # is used here to prevent the UI elements
                # from being displayed until the dataframe
                # is successfully populated
                plotDF = df(de_sols['y'], de_sols['t'])
                st.success("Solve successful! Plotting solution...")
                st.write(rf"**Numerical solution to** ${Tex}$")
                st.line_chart(data=plotDF)
            else:
                #this converts our sympy back into latex so it can be displayed again to the human eye so
                #accuracy can be confirmed
                st.write("Invalid differential equation: ")
                st.latex(Tex)
                st.write("please enter a valid differential equation")
    # pause further execution until the user
    # inputs another differential equation
    if st.button("Solve another differential equation"):
        st.rerun()
    else:
        st.stop()

st.write("""
# Infinitum

An interactive differential equations solver, developed by [Project Elara](https://elaraproject.org/). Enter a differential equation and Infinitum will numerically solve it for you! Source code is available on our [Codeberg repository](https://codeberg.org/elaraproject/elara-symbolic-ui/)

:warning: Be aware that the app currently only supports y(x) as the dependent variable. Also, the app is _highly experimental_, so if you encounter bugs please [report them to us](https://codeberg.org/elaraproject/elara-symbolic-ui/issues)!
""")

default_ode = r"\frac{dy}{dx} = y(1 - y)"
Tex = default_ode
Tex, _ = mathfield(title="Enter Equations Here", value=default_ode, mathml_preview=True, upright=False)

# Don't continue code execution until equation is specified
if not Tex:
    st.stop()

#Code for preliminary processing of the LaTeX
try:
    Tex = process_raw_text(Tex)
except LaTeXParsingError:
    st.error("Parsing of differential equation failed. Please check your inputted equation.")

#This is the code for the components letting the user set the bounds of the graph
lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)
stepSize = st.number_input(label="Enter Step Interval: ", value=0.01)

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex))
