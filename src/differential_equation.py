import streamlit as st
import sympy
import numpy as np
import re
from sympy.parsing.latex import parse_latex
from sympy import Derivative, Symbol, Function, Mul, symbols
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from polars import DataFrame
# Elara Symbolic imports must be at the end to avoid
# cyclical imports, which can cause an out-of-memory error
from elara_symbolic.cas import solve_ode
from elara_symbolic.numerical import RK4

class Differential_Equation:
    def __init__(self, strConstants, Tex):

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

        def solve_differential_equation(self, expr: sympy.Eq):
            print(expr)
            # parse the LaTeX provided by the user into a differential equation
            def replace_derivative(expr_to_fix):
                """
                    This accepts a sympy expression "expr_fragment" and then it basically parses in this expr fragment any subfragments that
                    can be identified as a falsely parsed higher order differential where sympy cannot parse with its latex parser but its parsing
                    is standardized enough we can create this function to do it for us. This function finds these recursively using that inner function.
                    It returns the correct expression rather than directly modifying the expression and inside our function to solve the differential equation it returns whether
                    the equation is a higher order differential equation though that functionality may soon be replaced.
                """
                
                def match_and_transform(expr_fragment):
                    # Check if the string representation matches our target
                    s = str(expr_fragment)
                    if re.fullmatch(r"\(d\*\*\d\*[a-z]\)\/\(d[a-z]\*\*\d\)", s):
                        num, denomFuncChar, numFuncChar = int(s[4]), s[6], s[11]
                        expr_fragment = Derivative(Function(denomFuncChar)(Symbol(numFuncChar)), Symbol(numFuncChar), num)
                    return expr_fragment

                # Run our inner function which recurisvely searches for incorrect sympy atoms
                return expr_to_fix.replace(lambda x: True, match_and_transform)
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

            # Fix up the equations where sympy improperly parsed derivatives
            expr = replace_derivative(expr)

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
            matchPattern = lambda node: isinstance(node, AppliedUndef) and node != dep_func
            while True:
                new_expr = expr.replace(matchPattern, fix_implicit_multiplication)
                if new_expr == expr: # Stop when no more changes are made
                    break
                expr = new_expr
            # Replace all y without a (x) with just a y(x)
            expr = expr.replace(Symbol(dep_func_name), dep_func)

            self._independent_var = indep_var
            self._dep_func = dep_func            
            self._expr = expr
        
        def process_constants(self, Constants):
            # create a dictionary of constants for the parsing
            parseConstants = { i : Symbol(i, constant=True, real=True) for i in Constants }
            # substitute parsed constants
            self._expr = self._expr.subs(parseConstants)
            return parseConstants

        
        unprocessed_sympy = parse_latex(process_raw_text(Tex), strict = False)
        self._latex = Tex
        solve_differential_equation(self, unprocessed_sympy)
        self._constants = process_constants(self, {i for i in strConstants})
    
    #Properties of the function that should not be modifiable parts of the class
    @property
    def constants(self) -> list[sympy.Symbol]: #returns a list of constants present in the function
        return self._constants
    @property
    def latex(self) -> str: #returns the original string value of the latex
        return self._latex
    @property
    def expr(self) -> sympy.Eq: #returns the processed value of the differential equation
        return self._expr
    @property
    def dep_func(self) -> sympy.Function: #returns the dependent function as a sympy Function object
        return self._dep_func
    @property
    def indep_var(self) -> sympy.Symbol: #returns the independent variable as a sympy Symbol
        return self._independent_var
    @property
    def order(self) -> int: # returns differential equation orderp
        return sympy.ode_order(self.expr, self.dep_func)
    
class Differential_Equation_Solution:
    def __init__(self, diffeq: Differential_Equation, upperRange: int, lowerRange: int, stepSize: float, Y0: list[float], solver: str):
        self._diffeq = diffeq
        self._range = (upperRange, lowerRange)
        self._step_size = stepSize
        self._Y0 = Y0
        self._solver = solver

        def solve_higher_order_diffeq(eq, dep_func) -> tuple[sympy.Eq, list[Symbol]]:
            """
            NOTE: Must change the name of this function it is bad.
            This function accepts an equation and a dependent function and substitutes each derivative of the dependent function except for
            the highest one (which stays a derivative of the substitution for the second highest function) with a substitute function for its
            order so y0 for dy/dx, y1 for d^2 y/dx^2, etc. It accepts eq, the equation to be substituted, and dep_func which is the dependent
            function so we know what function we are substituting. This function is useful for solving higher order diffeqs because it allows
            us to very easily get the coefficients of each derivative and solve using the method we use to solve higher order diffeqs. This
            returns the substituted equations and saves all the substituted equations in func_list.
            """
            func_list = []
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
            return (eq, func_list)
        def convert_diffeq_to_matrix(eq, func_list):
            """
            This function runs a reduction of order on an equation that has been ran through the function that substitutes derivatives for functions
            stored in func_list. This takes the coefficients and uses them to calculate a matrix that can be passed to RK4. This is the matrix a that
            is returned. It's a 2d numpy array of n-1 x n-1 where n is the order of the differential equation.
            """
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
            
        de_sols = {}
        constantPass = [(diffeq.constants[i], diffeq.constants.values()[i]) for i in diffeq.constants.keys()]
        if self._solver == "Base":
            de_sols = solve_ode(diffeq.expr, diffeq.dep_func, solver="trapezoidal", y0=Y0[0], t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
        elif self._solver == "leapfrog":
            if sympy.ode_order(diffeq.expr, diffeq.dep_func) == 1:
                de_sols = solve_ode(diffeq.expr, diffeq.dep_func, solver="leapfrog", y0=Y0[0], t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
            else:
                de_sols = solve_ode(diffeq.expr, diffeq.dep_func, solver="leapfrog", y0=Y0[0], v0=Y0[1], t_span=(lowerRange, upperRange), constants=constantPass, step_size=stepSize)
                de_sols['y'] = de_sols["x"][:, 0]
        elif solver == "RK4":
            temp = solve_higher_order_diffeq(diffeq.expr, diffeq.dep_func)
            higherOrderMatrix = convert_diffeq_to_matrix(*temp)
            print(higherOrderMatrix)
            if sympy.ode_order(diffeq.expr, diffeq.dep_func) == 1:
                f = sympy.lambdify((diffeq.indep_var, diffeq.dep_func), expr.rhs, modules="numpy")
            else:
                def f(t, y):
                    return higherOrderMatrix @ y
            x0 = np.array([1.0 for _ in Y0])
            de_sols = RK4(f, x0=x0, t_span=(lowerRange, upperRange), step_size=stepSize)
            if 'v' in de_sols:
                de_sols['y'] = de_sols["x"][:, 0]
        self._de_sols = de_sols

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
        self._plotDF = DataFrame({"x": de_sols['t']} | {kv: de_sols[kv].reshape(-1) for kv in functions})

    @property
    def plotDF(self): #returns the plotDF of a the given solution
        return self._plotDF

# diffeq1 = Differential_Equation("", r"\frac{d^3y}{d x^3}+\frac{d^2y}{dx^2}+\frac{d y}{d x}+y\left(x\right)=5")
# solution = Differential_Equation_Solution(diffeq1, 0, 1, .01, [1,1,1,1], "RK4")
# print(solution.plotDF)
