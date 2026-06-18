import streamlit as st
from sympy.parsing.latex import parse_latex
from sympy import Derivative, Symbol, Function, Mul, symbols
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from elara_symbolic.calculate import *
from st_mathlive import mathfield
import polars as pl
from PIL import Image
import re

#Load in the image from our library
icon = Image.open(r'src/favicon-144x144.png')
#This gives the page a title and icon in the browser so it is more identifiable
st.set_page_config(page_title="Infinitum", page_icon=icon)

st.toast("App loading...", icon="ℹ", duration="short")

# function for processing the mathlive user input into text that can be process by sympy
def process_raw_text(Tex: str):
    print(f"RAW TEX: {Tex}")
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

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float, leapfrogVal: bool):
    # parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    secondOrder = False

    def prepare_higher_order_diffeq(eq, dependent_symb, func_symbol):
        func_symbols = f"{func_symbol}_æ"
        symbol_string = f"{func_symbol} _ æ"
        tup = symbols(symbol_string, cls=Function)
        for i in range(0,len(tup)-1):
            substitute = Function(func_symbols[i])(dependent_symb)
            conv = Function(func_symbols[i+1])(dependent_symb)
            print(f"EQ: {eq}, {substitute}, {conv}\n")
            eq = eq.subs(substitute.diff(dependent_symb), conv)
        return eq

    def fixHigherOrderDiffeqs(layer):
        nonlocal expr
        nonlocal secondOrder
        curLayer = str(layer) # We make it a string so its easy to parse, the fragment of the equation we are checking
        #First exit case is if we reach the end and haven't found the target expression of a higher
        #order diffeq we exit
        if curLayer == "": 
            return False
        #if we have found the target equation, then we go ahead and substitute it in
        if re.fullmatch(r"\(d\*\*\d\*[a-z]\)\/\(d[a-z]\*\*\d\)", curLayer):
            #Get the function, degree, and dependent variable
            num = curLayer[4]
            denomFuncChar = curLayer[6]
            numFuncChar = curLayer[11]

            expr = expr.subs(layer, Derivative(Function(denomFuncChar)(Symbol(numFuncChar)),Symbol(numFuncChar),int(num)))
            secondOrder = True
            #let our loop know it should exit
            return True
        for i in layer.args:
            if fixHigherOrderDiffeqs(i):
                return True
        return False

    # create a dictionary of constants for the parsing
    parseConstants = { i : Symbol(i, constant=True, real=True) for i in constantValues.keys() }

    #Gotta do this before we find derivatives to find all the improperly parsed derivatives and convert them to properly parsed
    fixHigherOrderDiffeqs(expr)
    # Find all derivatives to figure out what the dependent function is
    derivatives = expr.atoms(Derivative)
    
    # get the functions that are not straight 
    dep_funcs = {deriv.args[0] for deriv in derivatives}
 
    dep_node = list(dep_funcs)[0]
    deriv = list(derivatives)[0]
    #if secondOrder:
    #    expr = prepare_higher_order_diffeq(expr, str(dep_node)[2], str(dep_node)[0])
    print(f"EXPR: {expr}")

    
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
    #This implements one algorithm for solving differential equations
    if not leapfrogVal:
        de_sols = solve_ode(expr, dep_func, y0=Y0, v0=0, t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
    #if there is only one arg we should not use the leapfrog algorithm
    elif len(dep_funcs) <= 1: 
        de_sols = solve_ode(expr, dep_func, y0=Y0, v0=0, t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
    #This implements a different algorithm, the leapfrog algorithm for higher order differential equations
    #elif leapfrogVal:
    #if True:
    #    de_sols = leapfrog(expr, x0=Y0, v0=None, t_span=(lowerRange, upperRange), fun_args=constantPass, step_size=stepSize, progress_bar=False, show_time_exec=False)

    return de_sols

# takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float, leapfrog: bool):
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
                    de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex, constantValues, Y0, leapfrog)
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
                plotDF = pl.DataFrame({"x": de_sols['t'], "y": de_sols['y']})
                print(plotDF.head(5))
                st.success("Solve successful! Plotting solution...")
                st.write(rf"**Numerical solution to** ${Tex}$")
                st.line_chart(data=plotDF, x='x', y='y')
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
# Elara-symbolic UI

An interactive differential equations solver, developed by [Project Elara](https://elaraproject.org/). Enter a differential equation and Infinitum will numerically solve it for you! Source code is available on our [Codeberg repository](https://codeberg.org/elaraproject/elara-symbolic-ui/)

:warning: Be aware that the app currently only supports y(x) as the dependent variable. Also, the app is _highly experimental_, so if you encounter bugs please [report them to us](https://codeberg.org/elaraproject/elara-symbolic-ui/issues)!
Currently being developed...
""")

# Code for preliminary processing of the LaTeX
Tex, _ = mathfield(title="Enter Equations Here", value=r"\frac{dy}{dx} = y(1 - y)", mathml_preview=True, upright=False)
Tex = process_raw_text(Tex) # Make sure to actually call your processing function!

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
Y0 = st.number_input(label="Enter Y0 of the Differential Equation: ", value=0.5) # Set the initial condition
selected_constants = st.selectbox(label="Enter The Solving Method You Wish To Use Here:", options=["Base", "Leapfrog"])

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0, True if selected_constants == "Leapfrog" else False))