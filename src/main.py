import streamlit as st
from sympy.parsing.latex import parse_latex
import sympy
from sympy import Derivative, Symbol, Function, Mul, symbols
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from elara_symbolic.cas import *
import polars as pl
from PIL import Image
import numpy as np
import re
import json
from pathlib import Path
from datetime import datetime

from mathquill_component import mathquill_input

# History file storage
HISTORY_FILE = Path.home() / ".infinitum_history.json"

def load_history():
    """Load calculation history from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_history(history):
    """Save calculation history to disk."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        st.warning(f"Could not save history: {e}")

def add_to_history(equation, constants, lower_range, upper_range, step_size, y0, solver):
    """Add a calculation to history."""
    history = load_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "equation": equation,
        "constants": constants,
        "lower_range": lower_range,
        "upper_range": upper_range,
        "step_size": step_size,
        "y0": y0,
        "solver": solver
    }
    history.append(entry)
    # Keep only last 50 entries to avoid huge file
    if len(history) > 50:
        history = history[-50:]
    save_history(history)

if "app_loaded" not in st.session_state:
    st.toast("App loading...", icon="ℹ", duration=2)
    st.session_state["app_loaded"] = True
    st.session_state["calculation_history"] = load_history()

def process_raw_text(Tex: str):
    r"""Implements a method for processing a raw user input string
    into a string readable by the sympy LaTeX parser.

    Parameters
    ----------
    Tex : string
    Accepts the user's input string to be processed

    Examples
    ---------
    >>> Processed_tex = process_raw_text($$ \frac{\mathrm{d}y}{\mathrm{d}x}=y(1-y) $$)
        For this input it would remove the \mathrm tags to allow the equation to be parsed by
        sympy's equation solver.
    
    """
    # we need this here because the mathfield often processes a simple d as this differentialD
    # which confuses sympy so we replace that with the simple d.
    newTex = Tex.replace(r"\differentialD", "d")
    # parse for upright equations TODO: Remove the option for upright equations
    if (r"\mathrm{" in newTex):
        newTex = newTex[8:-1]

    return newTex

#this finds and fixes implicit multiplication like y(1-y) rather than converting them to function calls
def fix_implicit_multiplication(node):
    """
    This function is quite simple. It is a function that takes a given unapplied def falsely parsed by sympy like y(1-y) which may be
    parsed as a function but we assume in this code is always y*(1-y) and simply returns y*(1-y) or just node with the proper expression.
    """
    func_name = node.func.__name__
    args = node.args
            
    # convert the function call to multiplication (e.g., y(1-y) -> y * (1-y))
    if len(args) == 1:
        return Symbol(func_name) * args[0]
    else:
        return Symbol(func_name) * Mul(*args)

def solve_differential_equation(upperRange: int, lowerRange: int, stepSize: float, Tex: str, constantValues: dict, Y0: float, solver: str):
    r"""Implements a method for processing a raw user input string
    into a string readable by the sympy LaTeX parser.

    Parameters
    ----------
    upperRange : int
    Upper range you want to be calculated on the graph

    lowerRange : int
    Lower range you want to be calculated on the graph, should be less than upper range

    StepSize : float
    What you want the interval of solve_ode to calculate on so for example you want the difference
    between steps to be .01 such that it calculates the y for y=.01,.02,.03,etc.. you input .01

    Tex: str
    This is simply the LaTeX of the expression to be parsed

    constantValues: dict
    Inputs the values of constants that the user inputs so the program knows which symbols in the sympy
    to pass to solve_ode as constants.

    Y0: float
    A list of initial conditions for each derivative so if y0 = 1 and y0' = 2 and y0'' = 3 then
    this input is [1,2,3] but if its a first order diffeq this input is merely [y0]

    solver: str

    Examples
    ---------
    >>> Processed_tex = process_raw_text($$ \frac{\mathrm{d}y}{\mathrm{d}x}=y(1-y) $$)
        For this input it would remove the \mathrm tags to allow the equation to be parsed by
        sympy's equation solver.
    
    """
    # parse the LaTeX provided by the user into a differential equation
    expr = parse_latex(Tex, strict=False)
    higherOrder = False

    func_list = []
    def solve_higher_order_diffeq(eq, dep_func):
        """
        NOTE: Must change the name of this function it is bad.
        This function accepts an equation and a dependent function and substitutes each derivative of the dependent function except for
        the highest one (which stays a derivative of the substitution for the second highest function) with a substitute function for its
        order so y0 for dy/dx, y1 for d^2 y/dx^2, etc. It accepts eq, the equation to be substituted, and dep_func which is the dependent
        function so we know what function we are substituting. This function is useful for solving higher order diffeqs because it allows
        us to very easily get the coefficients of each derivative and solve using the method we use to solve higher order diffeqs. This
        returns the substituted equations and saves all the substituted equations in func_list.
        """
        nonlocal func_list
        dependent_symb = dep_func.free_symbols.pop()
        base_func_name = str(dep_func)[0]
        order = sympy.ode_order(eq, dep_func)
        substitution_funcs = sympy.symbols(f"{base_func_name}0:{order-1}", cls=sympy.Function)
        substitution_funcs = [dep_func] + [f(dependent_symb) for f in substitution_funcs]
        for i in range(0,order-1):
            substitute = substitution_funcs[i]
            conv = substitution_funcs[i+1]
            func_list.append(substitute)
            eq = eq.subs(substitute.diff(dependent_symb), conv)
        func_list.append(conv)
        eq = sympy.Eq(eq.lhs - eq.rhs, 0)
        return eq
    def convert_diffeq_to_matrix(eq):
        """
        This function runs a reduction of order on an equation that has been ran through the function that substitutes derivatives for functions
        stored in func_list. This takes the coefficients and uses them to calculate a matrix that can be passed to RK4. This is the matrix a that
        is returned. It's a 2d numpy array of n-1 x n-1 where n is the order of the differential equation.
        """
        nonlocal func_list
        func_list.append(eq.atoms(sympy.Derivative).pop())
        a = np.empty((len(func_list)-1,len(func_list)-1))
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


    def replace_derivative(expr_to_fix):
        """
            This accepts a sympy expression "expr_fragment" and then it basically parses in this expr fragment any subfragments that
            can be identified as a falsely parsed higher order differential where sympy cannot parse with its latex parser but its parsing
            is standardized enough we can create this function to do it for us. This function finds these recursively using that inner function.
            It returns the correct expression rather than directly modifying the expression and inside our function to solve the differential equation it returns whether
            the equation is a higher order differential equation though that functionality may soon be replaced.
        """
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
    if not derivatives:
        raise ValueError("No derivative term was detected in the equation.")
    
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
    matchPattern = lambda node: isinstance(node, AppliedUndef) and node != dep_func
    while True:
        new_expr = expr.replace(matchPattern, fix_implicit_multiplication)
        if new_expr == expr: # Stop when no more changes are made
            break
        expr = new_expr
    # Replace all y without a (x) with just a y(x)
    expr = expr.replace(Symbol(dep_func_name), dep_func)
    #now get the matrix for a higher order diffeq
    if higherOrder:
        modExpr = solve_higher_order_diffeq(expr, dep_func)
        higherOrderMatrix = convert_diffeq_to_matrix(modExpr)

    print(f"SOLVING: {expr}, DEP_FUNC: {dep_func}")

    # Now convert any standalone symbols of the letter of the function to the formal function
    expr = expr.subs(Symbol(dep_func_name), dep_func)

    # substitute parsed constants
    expr = expr.subs(parseConstants)
    
    constantPass = [(parseConstants[i], constantValues[i]) for i in constantValues.keys()]
    
    # solve the differential equation itself
    de_sols = {}
    #This implements one algorithm for solving differential equations
    if solver == "Base":
        de_sols = solve_ode(expr, dep_func, solver="trapezoidal", y0=Y0[0], t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
        print(de_sols)
    elif solver == "Leapfrog":
        if sympy.ode_order(expr, dep_func) == 1:
            de_sols = solve_ode(expr, dep_func, solver="leapfrog", y0=Y0[0], t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
        else:
            de_sols = solve_ode(expr, dep_func, solver="leapfrog", y0=Y0[0], v0=Y0[1], t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
        de_sols['y'] = de_sols["x"][:, 0]
    #RK4 may be specified for higher order diffeqs
    elif solver == "RK4":
        if sympy.ode_order(expr, dep_func) == 1:
            #deriv = deriv.replace(Symbol(dep_func_name), dep_func)
            #expr = sympy.eq(deriv, (expr.lhs+expr.rhs/(expr.lhs.coeff(deriv)+expr.rhs.coeff(deriv))) - deriv)
            f = sympy.lambdify((indep_var, dep_func), expr.rhs, modules="numpy")
        else:
            def f(t, y):
                return higherOrderMatrix @ y
        x0 = np.array([float(v) for v in Y0], dtype=float)
        de_sols = RK4(f, x0=x0, t_span=(lowerRange, upperRange), step_size=stepSize)
        if 'v' in de_sols:
            de_sols['y'] = de_sols["x"][:, 0]

    return de_sols

# takes the equation, and the bounds and produces a graph from it
def process_input_and_graph(upperRange: int, lowerRange: int, stepSize: int, Tex: str, constantValues: dict, Y0: str, solver: str):
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
                    de_sols = solve_differential_equation(upperRange, lowerRange, stepSize, Tex, constantValues, Y0, solver)
                except ValueError as e:
                    st.error(f"Solve unsuccessful: {str(e)}")
                solve_complete = True
            if de_sols:
                st.session_state["valid_equation"] = True
                functions = ['y']
                if len(de_sols) > 2:
                    for i in range(0,len(de_sols['v'][0])):
                        newKeyName = 'dy'+str(i)
                        de_sols[newKeyName] = de_sols['v'][:,i]
                        functions.append(newKeyName)
                    
                #create a dataframe containing the date and time and plot that dataframe
                # this dataframe can be pretty slow to
                # initialize though, so this
                # is used here to prevent the UI elements
                # from being displayed until the dataframe
                # is successfully populated
                plotDF = pl.DataFrame({"x": de_sols['t']} | {kv: de_sols[kv].reshape(-1) for kv in functions})
                st.session_state["ode_solution"] = plotDF
                st.session_state["functions"] = functions
                st.session_state["latex"] = Tex
                
                # Save to history
                add_to_history(Tex, constantValues, lowerRange, upperRange, stepSize, Y0, solver)
                st.session_state["calculation_history"] = load_history()
                
                print(plotDF.head(5))
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
        st.latex(st.session_state.get("latex", ""))
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

    :warning: Be aware that the app is _highly experimental_, so if you encounter bugs please [report them to us](https://codeberg.org/elaraproject/elara-symbolic-ui/issues)!
    """)

    # Display calculation history in sidebar
    with st.sidebar:
        st.markdown("### 📋 Calculation History")
        history = st.session_state.get("calculation_history", [])
        
        if history:
            if st.button("🗑️ Clear History", key="clear_history_btn"):
                save_history([])
                st.session_state["calculation_history"] = []
                st.rerun()
            
            st.markdown("---")
            st.markdown("**Recent Calculations:**")
            
            # Display history in reverse order (newest first)
            for idx, entry in enumerate(reversed(history[-10:])):
                equation = entry.get("equation", "Unknown")
                timestamp = entry.get("timestamp", "")
                
                # Format timestamp
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%m/%d %H:%M")
                    except:
                        time_str = "Unknown"
                else:
                    time_str = "Unknown"
                
                # Create a button to load this calculation
                if st.button(
                    f"📌 {equation[:30]}...\n_{time_str}_",
                    key=f"history_btn_{len(history)-1-idx}",
                    use_container_width=True
                ):
                    # Load the calculation
                    st.session_state["diffeq"] = equation
                    st.session_state["load_history_entry"] = entry
                    st.rerun()

    # Default ODE is the logistic equation
    default_ode = r"\frac{dy}{dx} = y(1 - y)"
    Tex = default_ode

    equation_to_load = st.session_state["diffeq"] if "diffeq" \
                        in st.session_state else default_ode

    # Code for preliminary processing of the LaTeX
    try:
        Tex = mathquill_input(
            label="Enter Equations Here",
            value=equation_to_load,
            key="mq_input",
            placeholder=r"\frac{dy}{dx} = y(1-y)",
        )
    except Exception:
        # Hard fallback so equation input remains usable if the component fails.
        Tex = st.text_input(
            label="Enter Equations Here (LaTeX)",
            value=equation_to_load,
            key="mq_input_fallback",
        )
    # Pause execution if equation is not yet parsed
    if not Tex:
        st.stop()
    Tex = process_raw_text(Tex) # Make sure to actually call your processing function!

    # Remember the user's last solved differential
    # equation and load it
    #st.session_state["diffeq"] = Tex

    # Check if we should load a history entry
    if "load_history_entry" in st.session_state:
        entry = st.session_state["load_history_entry"]
        
        # Restore parameters from history
        selected_constants = list(entry.get("constants", {}).keys())
        constant_values = entry.get("constants", {})
        lowerRange = entry.get("lower_range", 0.0)
        upperRange = entry.get("upper_range", 1.0)
        stepSize = entry.get("step_size", 0.01)
        Y0 = entry.get("y0", "0.5")
        selected_constants_dropdown = entry.get("solver", "Base")
        
        # Clear the load flag
        del st.session_state["load_history_entry"]
        
        # Auto-solve the loaded calculation
        st.info("✨ Loaded from history! Solving...")
        process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0, selected_constants_dropdown)
        st.rerun()
    
    # code for selecting what will be a constant and setting the value of said constant
    selected_constants = st.multiselect(label="Enter List of Constants", options=list('abcdefghijklmnopqrstuvwxyz'))
    constant_values = {i : 0.0 for i in selected_constants}
    for letter in selected_constants:
        constant_values[letter] = st.number_input(label=f"enter constant value for {letter}: ", value=0.0)

    # This is the code for the components letting the user set the bounds of the graph
    lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
    upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)
    stepSize = st.number_input(label="Enter Step Interval: ", value=0.01)
    Y0 = st.text_input(label="Enter Y0 of the Differential Equation as a Comma Separated List (e.g. 0,1,2,3...): ", value="0.5") # Set the initial condition
    selected_constants = st.selectbox(label="Enter The Solving Method You Wish To Use Here:", options=["Base", "Leapfrog", "RK4"])

    st.button(label="Solve Differential Equation", on_click=lambda: process_input_and_graph(upperRange, lowerRange, stepSize, Tex, constant_values, Y0, selected_constants))
