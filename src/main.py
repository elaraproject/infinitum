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

Tex, MathML = mathfield(title="Enter Equations Here")

#NOTE: BUG EXISTS WHERE IT MAY HAVE TROUBLE PARSING CERTAIN UPRIGHT EQUATIONS, DO NOT CHECK THAT BOX
expr = parse_latex(Tex, strict=True)

#for future use developing the differential equation solving
#k = Symbol("k", constant=True, real=True)
#de = solve_ode(expr, y, y0=0.2, t_span=(0, 4), constants=[(k, 1.0)], step_size=0.01)

#this converts our sympy back into latex so it can be displayed again to the human eye so
#accuracy can be confirmed
newTex = latex(expr)

#code for future stuff with graphing
#st.line_chart(de)

#output the latex so we know that sympy has properly processed the equation
st.write(newTex)