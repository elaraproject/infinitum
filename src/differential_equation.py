import streamlit as st
from sympy.parsing.latex import parse_latex
import sympy
from sympy import Derivative, Symbol, Function, Mul, symbols
from sympy.core.function import AppliedUndef # Crucial for finding mistaken function calls
from st_mathlive import mathfield
import polars as pl
from elara_symbolic.cas import *
from PIL import Image
import numpy as np
import re

class Differential_Equation:
    def __init__(self, strConstants: list[str], Tex: str):

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

        def process_differential_equation(self, expr: sympy.Eq):
            r"""Implements a method for processing sympy's 'bad' differential equation parsing into
            a differential equation that can be solved by elara_symbolic using situation-specific
            parsing for ordinary diffferential equations

            Parameters
            ----------
            expr : sympy.Eq
            Accepts the user's input equation to be parsed

            Examples
            ---------
            >>> processed_expr = process_raw_text(r"\frac{\mathrm{d}y}{\mathrm{d}x}=y(1-y)")
                This sets the properties of self._indep_var to x, self._dep_var to y(x), 
                and self._expr to dy/dx=y(1-y)
            
            """
            def replace_derivative(expr_to_fix):
                r"""This function manages to replace any derivatives at a higher order than dy/dx
                to d^2 y /dx^2 such that d is not a constant or symbol but rather it a d^2 y. So
                for higher order diffeqs it fixes sympy's poor processing of the differential.

                Parameters
                ----------
                expr_to_fix : sympy.Eq
                Accepts the user's equation where the differentials should be processed correcctly

                Examples
                ---------
                >>> processed_expr = process_raw_text(sympy.Eq(d^2y/dx^2 = y(1-y)))
                    This converts the it such that the d^2y/dx^2 is a differential rather than where d is a symbol (d^2)*(y)
                
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
            def fix_implicit_multiplication(node: sympy.Eq):
                r"""This function is a recursive function that converts function calls that imply implicit multiplication
                to a a multiplicative expression

                Parameters
                ----------
                node : sympy.Eq
                Accepts the node on which to run the recursive function and check for implicit multiplication.

                Examples
                ---------
                >>> processed_expr = process_raw_text(sympy.Eq(y(y(y(y)-1)))))
                    this converts the given expression to y*(interior) and does a recursive call to convert the interior of y(y(y(y)-1))
                    such that it convers all the calls that are not y(indep_var) to multiplication because we've chosen not to support that
                    for this ODE class.
                
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
        
        def process_constants(self, Constants: list[str]) -> dict[str : sympy.Symbol]:
            # create a dictionary of constants for the parsing
            parseConstants = { i : Symbol(i, constant=True, real=True) for i in Constants }
            # substitute parsed constants
            self._expr = self._expr.subs(parseConstants)
            return parseConstants

        #Begin by processing the latex so that it can be processed correctly by 
        unprocessed_sympy = parse_latex(process_raw_text(Tex), strict = False)
        #Save the string we created
        self._latex = Tex
        #This function here saves the differential equation's information as class variables that can be accessed by properties
        process_differential_equation(self, unprocessed_sympy)
        #Finish processing the differential equations by substituting in the constants and save the list of constants as a property
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

        def substitute_higher_differentials(eq: sympy.Eq, dep_func: sympy.Function) -> tuple[sympy.Eq, list[Symbol]]:
            r"""This accepts a higher order differential equation such that it substitutes higher order differential equations
            with substitute functions that allow for a reduction of order to be performed. It returns a list of the substitution
            functions and the substituted differential equation itself.

                Parameters
                ----------
                eq : sympy.Eq
                Accepts the higher order differential equation such that we can substitute the higher order differentials.

                node : sympy.Function
                Accepts the dependent function so that the order of the differential equation can be deduced.

                Examples
                ---------
                >>> processed_expr = substitute_higher_differentials(sympy.Eq(d^2y/dx^2 + d^3y/dx^3 + dy/dx + y(x) = 5))
                    This substitutes the higher order differentials except for the highest order ones with characters so
                    it becomes dy1/dx + y1(x) + y0(x) + y(x) = 5. It also returns the funclist of [y(x),y0(x),y1(x),dy2/dx]
                
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
        def convert_diffeq_to_matrix(eq: sympy.Eq, func_list: list[sympy.Function]) -> np.array:
            r"""This function creates a reduction of order matrix so that a function can be solved even at a higher
            order. Basically an array of coefficients to be solved in a way that RK4 can tolerate.

                Parameters
                ----------
                eq : sympy.Eq
                Accepts a substituted equation where the higher order differentials have been replaced by functions.

                func_list : list[sympy.Function]
                accepts a list of dependent differential functions to iterate through for finding the coefficients.

                Examples
                ---------
                >>> processed_expr = substitute_higher_differentials(sympy.Eq(dy1/dx + y1(x) + y0(x) + y(x) = 5), [y(x),y0(x),y1(x),dy2/dx])
                    This creates a matrix of floats such that the coefficients in front of each get turned into an array of order x order
                    coefficients. it returns this array.
                
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
            temp = substitute_higher_differentials(diffeq.expr, diffeq.dep_func)
            higherOrderMatrix = convert_diffeq_to_matrix(*temp)
            if sympy.ode_order(diffeq.expr, diffeq.dep_func) == 1:
                f = sympy.lambdify((diffeq.indep_var, diffeq.dep_func), expr.rhs, modules="numpy")
            else:
                def f(t, y):
                    return higherOrderMatrix @ y
            x0 = np.array([_ for _ in Y0])
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
        self._functions = functions
        self._plotDF = pl.DataFrame({"x": de_sols['t']} | {kv: de_sols[kv].reshape(-1) for kv in functions})

    @property
    def plotDF(self): #returns the plotDF of a the given solution
        return self._plotDF
    @property
    def functions(self): #returns a list of the given derivatives
        return self._functions