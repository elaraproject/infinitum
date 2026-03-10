import streamlit as st
from bs4 import BeautifulSoup
from sympy.parsing.latex import parse_latex
from elara_symbolic.calculate import *
from st_mathlive import mathfield

st.write("""
# Elara-symbolic UI

Currently being developed...
""")

Tex, MathML = mathfield(title="Enter Equations Here", value=r"\frac{dy}{dx} = y(1 - y)", mathml_preview=True, upright=False)

#NOTE: BUG EXISTS WHERE IT MAY HAVE TROUBLE PARSING CERTAIN UPRIGHT EQUATIONS, DO NOT CHECK THAT BOX
Tex = Tex.replace(r"\differentialD", "d")
#parse for upright equations TODO: Remove the option for upright equations
if (Tex[0:8] == "\mathrm{"):
    Tex = Tex[8:-1]
print(Tex)
expr = parse_latex(Tex, strict=False)

# define your unknown function
# to be solved for
# here it is y(x)
y = Function("y")(x)

lowerRange = st.number_input(label="Enter Lower Number Bound: ", value=0.0)
upperRange = st.number_input(label="Enter Upper Number Bound: ", value=1.0)


if upperRange <= lowerRange:
    st.write("Unable to display equation: lower bound is greater than or equal to lower bound ")
else:
    # define a dummy constant for solving the differential equation
    k = Symbol("k", constant=True, real=True)
    #solve the differential equation itself
    de_sols = solve_ode(expr, y, y0=0.0, t_span=(lowerRange, upperRange), constants=[(k, 1.0)], step_size=0.01)
    if type(de_sols) != type(None):
        st.line_chart(de_sols['y'])
    else:
        #this converts our sympy back into latex so it can be displayed again to the human eye so
        #accuracy can be confirmed
        newTex = latex(expr)
        st.write("Invalid differential equation: ")
        st.latex(newTex)
        st.write("please enter a valid differential equation")
