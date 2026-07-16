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

class Differential_Equation:
    def __init__(self, Constants, Tex):
        self._latex = Tex
        self._constants = Constants

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
        
        unprocessed_sympy = process_raw_text(self.Tex)

        def solve_differential_equation(Tex: sympy.Eq):
            # parse the LaTeX provided by the user into a differential equation
            func_list = []
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
        solve_differential_equation(self, Tex)
    
    #Properties of the function that should not be modifiable parts of the class
    @property
    def constants(self):
        return self._constants
    @property
    def latex(self):
        return self._latex
    @property
    def expr(self):
        return self._expr
    @property
    def dep_func(self):
        return self._dep_func
    @property
    def indep_var(self):
        return self._independent_var