import streamlit as st
from sympy.parsing.latex import parse_latex
from sympy import Derivative, Symbol, Function, Mul
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from elara_symbolic.calculate import *
from st_mathlive import mathfield
from pandas import DataFrame as df

# function for processing the mathlive user input into text that can be process by sympy
def process_raw_text(Tex: str):
    # we need this here because the mathfield often processes a simple d as this differentialD
    # which confuses sympy so we replace that with the simple d.
    newTex = Tex.replace(r"\differentialD", "d")
    # parse for upright equations TODO: Remove the option for upright equations
    if (r"\mathrm{" in newTex):
        newTex = newTex[8:-1]

    return newTex

#this finds and fixes implicit multiplication like y(1-y) rather than converting them to function calls
def fix_implicit_multiplication(node):
    if isinstance(node, AppliedUndef):
        func_name = node.func.__name__
        args = node.args
            
        # 1. Preserve the dependent variable if it's evaluated exactly at the independent variable (e.g., y(x))
        if func_name == dep_func_name and len(args) == 1 and args[0] == indep_var:
            return node
            
        # 2. Otherwise, convert the function call to multiplication (e.g., y(1-y) -> y * (1-y))
        if len(args) == 1:
            return Symbol(func_name) * args[0]
        else:
            return Symbol(func_name) * Mul(*args)
    return node

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float):
    # parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    
    # create a dictionary of constants for the parsing
    parseConstants = { i : Symbol(i, constant=True, real=True) for i in constantValues.keys() }

    # Find all derivatives to figure out what the dependent function is
    derivatives = expr.atoms(Derivative)
    
    # get the functions that are not straight 
    dep_funcs = {deriv.args[0] for deriv in derivatives}
    
    dep_node = list(dep_funcs)[0]
    deriv = list(derivatives)[0]
    
    # SymPy sometimes parses the derivative dependent var as a Symbol ('y'), sometimes as a Function ('y(x)')
    if isinstance(dep_node, AppliedUndef):
        dep_func_name = dep_node.func.__name__
        indep_var = dep_node.args[0]
        dep_func = dep_node
    else:
        dep_func_name = dep_node.name
        # Extract independent variable from derivative args (e.g., x from dy/dx)
        v = deriv.args[1]
        indep_var = v[0] if type(v).__name__ in ['tuple', 'Tuple'] else v
        # Force a proper y(x) representation for ODE solvers
        dep_func = Function(dep_func_name)(indep_var)

    # We use a while loop to handle nested multiplications like k(y(1-y)) smoothly.
    while True:
        new_expr = expr.replace(AppliedUndef, fix_implicit_multiplication)
        if new_expr == expr: # Stop when no more changes are made
            break
        expr = new_expr

    # Now convert any standalone symbols of the letter of the function to the formal function
    expr = expr.subs(Symbol(dep_func_name), dep_func)

    # substitute parsed constants
    expr = expr.subs(parseConstants)
    
    # define a dummy constant for solving the differential equation
    k = Symbol("k", constant=True, real=True)
    
    constantPass = [(parseConstants[i], constantValues[i]) for i in constantValues.keys()] if len(constantValues) > 0 else [(k, 1.0)]
    
    # solve the differential equation itself
    de_sols = solve_ode(expr, dep_func, y0=Y0, t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)

    return de_sols

# takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float):
    if upperRange <= lowerRange:
        st.write("Unable to display equation: lower bound is greater than or equal to upper bound.")
    else:
        try:
            de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex, constantValues, Y0)
            if de_sols is not None:
                # create a dataframe containing the date and time and plot that dataframe
                plotDF = df(de_sols['y'], de_sols['t']) 
                st.line_chart(data=plotDF)
            else:
                st.write("Invalid differential equation: ")
                st.latex(Tex)
                st.write("please enter a valid differential equation")
        except Exception as e:
            st.error(f"Error solving equation: {e}")

st.write("""
# Elara-symbolic UI

Currently being developed...
""")

# Code for preliminary processing of the LaTeX
Tex, _ = mathfield(title="Enter Equations Here", value=r"\frac{dy}{dx} = y(1 - y)", mathml_preview=True, upright=False)
Tex = process_raw_text(Tex) # Make sure to actually call your processing function!

# code for selecting what will be a constant and setting the value of said constant
selected_constants = st.multiselect(label="Enter List of Constants", options=list('abcdefghijklmnopqrstuvwxyz'))
constant_values = {i : 0.0 for i in selected_constants}
for letter in selected_constants:
    constant_values[letter] = st.number_input(label=f"enter constant value for {letter}: ", value=0.0)

# This is the code for the components letting the user set the bounds of the graph
lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)
stepSize = st.number_input(label="Enter Step Interval: ", value=0.01)
Y0 = st.number_input(label="Enter Y0 of the Differential Equation: ", value=0.5) # Set the initial condition

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0))