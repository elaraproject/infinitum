"""
Symbolic Function Plotter Module

Allows users to define symbolic expressions and plot them with:
- Automatic variable/constant detection
- Interactive sliders for constants
- Real-time plot updates
- Support for Cartesian and polar coordinates
"""

import streamlit as st
import numpy as np
import sympy as sp
from sympy import symbols, sympify, lambdify, pi, cos, sin, sqrt
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set


class SymbolicPlotterError(Exception):
    """Base exception for symbolic plotter failures."""


class PlotGenerationError(SymbolicPlotterError):
    """Raised when a plot cannot be generated from the current expression."""


class SymbolicFunctionPlotter:
    """Parse and plot symbolic mathematical expressions."""
    
    def __init__(self):
        self.expression = None
        self.variables = set()
        self.constants = {}
        self.free_symbols = set()
        
    def parse_expression(self, expr_str: str) -> bool:
        """
        Parse a symbolic expression string.
        
        Args:
            expr_str: Expression string (e.g., "y = sin(x) + a*x")
            
        Returns:
            bool: True if parsing successful, False otherwise
        """
        try:
            # Remove "y = " if present
            if "=" in expr_str:
                expr_str = expr_str.split("=", 1)[1].strip()
            
            # Parse the expression
            self.expression = sympify(expr_str, transformations='all')
            
            # Get all free symbols
            self.free_symbols = self.expression.free_symbols
            
            # Classify symbols as variables or constants
            # Assume single letter symbols near start of alphabet are variables
            # (x, y, t, etc.) and others are constants
            variable_names = {'x', 'y', 't', 'u', 'v', 'w', 'theta', 'r', 'phi'}
            
            self.variables = {str(s) for s in self.free_symbols if str(s) in variable_names}
            self.constants = {str(s): 1.0 for s in self.free_symbols if str(s) not in self.variables}
            
            # If no variables detected, assume x is the variable
            if not self.variables:
                self.variables = {'x'}
                self.constants = {str(s): 1.0 for s in self.free_symbols if str(s) != 'x'}
            
            return True
        except Exception as e:
            st.error(f"Failed to parse expression: {str(e)}")
            return False
    
    def get_constants(self) -> Dict[str, float]:
        """Get all detected constants."""
        return self.constants
    
    def get_variables(self) -> Set[str]:
        """Get all detected variables."""
        return self.variables
    
    def generate_cartesian_plot(self, x_range: Tuple[float, float] = (-10, 10), 
                                num_points: int = 500) -> plt.Figure:
        """
        Generate a Cartesian plot for y = f(x).
        
        Args:
            x_range: Range for x-axis (min, max)
            num_points: Number of points to plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Create lambdified function
            x_sym = sp.Symbol('x')
            
            # Substitute constant values
            expr_with_constants = self.expression
            for const_name, const_value in self.constants.items():
                const_sym = sp.Symbol(const_name)
                expr_with_constants = expr_with_constants.subs(const_sym, const_value)
            
            # Create the function
            f = lambdify(x_sym, expr_with_constants, modules=['numpy'])
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], num_points)
            
            # Calculate y values
            try:
                y_vals = f(x_vals)
            except:
                y_vals = np.array([float(f(x)) for x in x_vals])
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, 'b-', linewidth=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'y = {str(self.expression)}')
            
            return fig
            
        except Exception as e:
            raise PlotGenerationError(f"Error generating plot: {str(e)}") from e
    
    def generate_polar_plot(self, theta_range: Tuple[float, float] = (0, 2*np.pi),
                           num_points: int = 500) -> plt.Figure:
        """
        Generate a polar plot for r = f(theta).
        
        Args:
            theta_range: Range for theta (min, max) in radians
            num_points: Number of points to plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Create lambdified function with theta as variable
            theta_sym = sp.Symbol('theta')
            
            # Substitute constant values
            expr_with_constants = self.expression
            for const_name, const_value in self.constants.items():
                const_sym = sp.Symbol(const_name)
                expr_with_constants = expr_with_constants.subs(const_sym, const_value)
            
            # Replace x/y with theta if needed
            x_sym = sp.Symbol('x')
            expr_with_constants = expr_with_constants.subs(x_sym, theta_sym)
            
            # Create the function
            f = lambdify(theta_sym, expr_with_constants, modules=['numpy'])
            
            # Generate theta values
            theta_vals = np.linspace(theta_range[0], theta_range[1], num_points)
            
            # Calculate r values
            try:
                r_vals = f(theta_vals)
            except:
                r_vals = np.array([float(f(t)) for t in theta_vals])
            
            # Create polar plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            ax.plot(theta_vals, np.abs(r_vals), 'b-', linewidth=2)
            ax.set_title(f'r = {str(self.expression)}')
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            raise PlotGenerationError(f"Error generating polar plot: {str(e)}") from e


def create_plotter_ui():
    """Create the symbolic function plotter UI."""
    
    st.markdown("## 📊 Symbolic Function Plotter")
    st.markdown("""
    Define a symbolic expression and visualize it interactively.
    
    **Examples:**
    - `sin(x) + a*x` - Linear + sine combination
    - `a*x**2 + b*x + c` - Parabola
    - `exp(-a*x)*cos(b*x)` - Damped oscillation (polar: `a*exp(-b*theta)`)
    """)
    
    # Initialize plotter if not in session state
    if "plotter" not in st.session_state:
        st.session_state.plotter = SymbolicFunctionPlotter()
    
    plotter = st.session_state.plotter
    
    # Expression input
    expr_input = st.text_input(
        "Enter expression (e.g., sin(x) + a*x):",
        value="",
        placeholder="e.g., sin(x) + a*x"
    )
    
    if expr_input:
        if plotter.parse_expression(expr_input):
            st.session_state.expr_valid = True
            
            # Display detected variables and constants
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📍 Variables: {', '.join(sorted(plotter.variables))}")
            with col2:
                st.info(f"⚙️ Constants: {', '.join(sorted(plotter.constants.keys())) if plotter.constants else 'None'}")
            
            # Create sliders for constants
            if plotter.constants:
                st.markdown("### Adjust Constants:")
                cols = st.columns(min(len(plotter.constants), 3))
                
                for idx, (const_name, const_value) in enumerate(plotter.constants.items()):
                    col = cols[idx % len(cols)]
                    with col:
                        plotter.constants[const_name] = st.slider(
                            f"{const_name}",
                            min_value=-10.0,
                            max_value=10.0,
                            value=float(const_value),
                            step=0.1,
                            key=f"slider_{const_name}"
                        )
            
            # Plot type selection
            plot_type = st.radio(
                "Plot Type:",
                ["Cartesian (y=f(x))", "Polar (r=f(θ))"],
                horizontal=True
            )
            
            # Plot generation
            if st.button("📈 Generate Plot"):
                with st.spinner("Generating plot..."):
                    try:
                        if plot_type == "Cartesian (y=f(x))":
                            x_min = st.slider("X min", -50.0, 50.0, -10.0, key="x_min")
                            x_max = st.slider("X max", -50.0, 50.0, 10.0, key="x_max")
                            fig = plotter.generate_cartesian_plot(x_range=(x_min, x_max))
                        else:
                            fig = plotter.generate_polar_plot()

                        st.pyplot(fig)
                        st.session_state.last_plot = fig
                    except PlotGenerationError as exc:
                        st.error(str(exc))
        else:
            st.error("❌ Invalid expression. Please check the syntax.")
