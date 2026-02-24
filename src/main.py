import streamlit as st
from bs4 import BeautifulSoup
from sympy.parsing.latex import parse_latex
from elara_symbolic.calculate import *

st.write("""
# Elara-symbolic UI

Currently being developed...
""")

import streamlit as st
from st_mathlive import mathfield

Tex, MathML = mathfield(title="Enter Equations Here", upright=False, mathml_preview=False)

#NOTE: BUG EXISTS WHERE IT MAY HAVE TROUBLE PARSING CERTAIN UPRIGHT EQUATIONS, DO NOT CHECK THAT BOX
Tex = Tex.replace("\differentialD", "d")
print("RIGHT HERE" + Tex)
expr = parse_latex(Tex, strict=True)

# define your unknown function
# to be solved for
# here it is y(x)
y = Function("y")(x)

# define a dummy constant for solving the differential equation
k = Symbol("k", constant=True, real=True)

# define the differential equation with SymPy
diffeq = Eq(D(y, x) - k*y*(1-y), 0)

range = st.slider("Enter Differential Equation Range:", 0, 130, 25)

#output the latex so we know that sympy has properly processed the equation and graph it for the user
de_sols = solve_ode(expr, y, y0=0.0, t_span=(0, range), constants=[(k, 1.0)], step_size=0.01)
if type(de_sols) != type(None):
    st.line_chart(de_sols['y'])
else:
    #this converts our sympy back into latex so it can be displayed again to the human eye so
    #accuracy can be confirmed
    newTex = latex(expr)
    st.write("Invalid differential equation: ")
    st.latex(newTex)
    st.write("please enter a valid differential equation")
