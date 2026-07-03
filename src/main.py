import streamlit as st
from sympy.parsing.latex import parse_latex
import sympy
from sympy import Derivative, Symbol, Function, Mul, symbols
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from elara_symbolic.cas import *
from st_mathlive import mathfield
import polars as pl
from PIL import Image
import numpy as np
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
    func_name = node.func.__name__
    args = node.args
            
    # convert the function call to multiplication (e.g., y(1-y) -> y * (1-y))
    if len(args) == 1:
        return Symbol(func_name) * args[0]
    else:
        return Symbol(func_name) * Mul(*args)

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float, solver: str):
    # parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    higherOrder = False

    func_list = []
    def solve_higher_order_diffeq(eq, dependent_symb, func_symbol):
        nonlocal func_list
        func_symbols = f"{func_symbol}_æ"
        symbol_string = f"{func_symbol} _ æ"
        tup = sympy.symbols(symbol_string, cls=sympy.Function)
        for i in range(0,len(tup)-1):
            #sympy.wild? dummy?
            substitute = sympy.Function(func_symbols[i])(dependent_symb)
            conv = sympy.Function(func_symbols[i+1])(dependent_symb)
            func_list.append(substitute)
            eq = eq.subs(substitute.diff(dependent_symb), conv)
        func_list.append(conv)
        eq = sympy.Eq(eq.lhs - eq.rhs, 0)
        return eq
    def convert_diffeq_to_matrix(eq):
        nonlocal func_list
        func_list.append(eq.atoms(sympy.Derivative).pop())
        a = np.empty((3,3))
        print(func_list)
        x = 0
        for i in func_list[1::]:
            coeffless_func = i
            coeff = eq.lhs.coeff(i)
            i = coeffless_func * coeff
            dummy = sympy.Eq(-eq.rhs + i, -eq.lhs + i)
            if coeffless_func == func_list[-1]:
                a[x] = np.array([dummy.rhs.coeff(z)/coeff for z in func_list if coeffless_func != z])
            else:
                a[x] = np.array([1 if z == x+1 else 0 for z in range(0,len(func_list)-1)])
            x += 1
        return a

    # This function basically corrects a weird thing done by sympy where it improperly parses the d as constant and basically
    # botches the parsing so this function for botched parsings and replaces them with the correct derivative
    def replace_derivative(expr_to_fix):
        nonlocal higherOrder
        
        def match_and_transform(expr_fragment):
            nonlocal higherOrder
            # Check if the string representation matches our target
            s = str(expr_fragment)
            if re.fullmatch(r"\(d\*\*\d\*[a-z]\)\/\(d[a-z]\*\*\d\)", s):
                num = int(s[4])
                if num > 2: higherOrder = True
                denomFuncChar = s[6]
                numFuncChar = s[11]
                expr_fragment = Derivative(Function(denomFuncChar)(Symbol(numFuncChar)), Symbol(numFuncChar), num)
            return expr_fragment

        # Run our inner function which recurisvely searches for incorrect sympy atoms
        return expr_to_fix.replace(lambda x: True, match_and_transform)

    # Fix up the equations where sympy improperly parsed derivatives
    expr = replace_derivative(expr)

    # create a dictionary of constants for the parsing
    parseConstants = { i : Symbol(i, constant=True, real=True) for i in constantValues.keys() }

    # Find all derivatives to figure out what the dependent function is
    derivatives = expr.atoms(Derivative)
    
    # get the functions that are not straight 
    dep_funcs = {deriv.args[0] for deriv in derivatives}
    dep_node = list(dep_funcs)[0]
    deriv = list(derivatives)[0]

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
    matchPattern = lambda node: isinstance(node, AppliedUndef) and node != dep_func
    while True:
        new_expr = expr.replace(matchPattern, fix_implicit_multiplication)
        if new_expr == expr: # Stop when no more changes are made
            break
        expr = new_expr
    #now get the matrix for a higher order diffeq
    if higherOrder:
        modExpr = solve_higher_order_diffeq(expr, indep_var.name, dep_func.func.__name__,)
        higherOrderMatrix = convert_diffeq_to_matrix(modExpr)
        print(higherOrderMatrix)
    # Replace all y without a (x) with just a y(x)
    expr = expr.replace(Symbol(dep_func_name), dep_func)

    print(f"SOLVING: {expr}, DEP_FUNC: {dep_func}")

    # Now convert any standalone symbols of the letter of the function to the formal function
    expr = expr.subs(Symbol(dep_func_name), dep_func)

    # substitute parsed constants
    expr = expr.subs(parseConstants)
    
    # define a dummy constant for solving the differential equation
    k = Symbol("k", constant=True, real=True)
    
    constantPass = [(parseConstants[i], constantValues[i]) for i in constantValues.keys()] if len(constantValues) > 0 else [(k, 1.0)]
    
    # solve the differential equation itself
    de_sols = {}
    #This implements one algorithm for solving differential equations
    if solver == "Base":
        de_sols = solve_ode(expr, dep_func, solver="trapezoidal", y0=Y0, v0=0, t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
    elif solver == "Leapfrog":
        de_sols = solve_ode(expr, dep_func, solver="leapfrog", y0=Y0, v0=0, t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
        de_sols['y'] = de_sols["v"][:, 0]
        print("SOLVED")
    #RK4 may be specified for higher order diffeqs
    elif solver == "RK4":
        def f(t, y):
            return higherOrderMatrix @ y
        x0 = np.array([1.0, 1.0, 1.0])
        de_sols = RK4(f, x0=x0, t_span=(lowerRange, upperRange), step_size=stepSize)
        if 'v' in de_sols:
            de_sols['y'] = de_sols["x"][:, 0]

    return de_sols

# takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: float, solver: str):
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
                    de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex, constantValues, Y0, solver)
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
selected_constants = st.selectbox(label="Enter The Solving Method You Wish To Use Here:", options=["Base", "Leapfrog", "RK4"])

st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0, selected_constants))
