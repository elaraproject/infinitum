import re

import numpy as np
import sympy
from sympy import Derivative, Function, Mul, Symbol
from sympy.core.function import AppliedUndef
from sympy.parsing.latex import parse_latex

from elara_symbolic.cas import RK4, solve_ode


def _process_raw_text(tex: str) -> str:
    """Normalize raw LaTeX from the editor before SymPy parsing."""
    normalized = tex.replace(r"\differentialD", "d")
    if r"\mathrm{" in normalized:
        normalized = normalized[8:-1]
    return normalized


def _fix_implicit_multiplication(node):
    """Convert implicit call-like syntax y(1-y) into multiplication."""
    func_name = node.func.__name__
    args = node.args
    if len(args) == 1:
        return Symbol(func_name) * args[0]
    return Symbol(func_name) * Mul(*args)


def _replace_derivative(expr_to_fix):
    """Fix malformed higher-order derivative tokens from LaTeX parsing."""

    def match_and_transform(expr_fragment):
        s = str(expr_fragment)
        if re.fullmatch(r"\(d\*\*\d\*[a-z]\)\/\(d[a-z]\*\*\d\)", s):
            num, denom_func_char, num_func_char = int(s[4]), s[6], s[11]
            return Derivative(
                Function(denom_func_char)(Symbol(num_func_char)),
                Symbol(num_func_char),
                num,
            )
        return expr_fragment

    return expr_to_fix.replace(lambda x: True, match_and_transform)


def _solve_higher_order_diffeq(eq, dep_func):
    """Reduce a higher-order ODE to first-order system form."""
    func_list = []
    dependent_symb = dep_func.free_symbols.pop()
    base_func_name = str(dep_func)[0]
    order = sympy.ode_order(eq, dep_func)
    substitution_funcs = sympy.symbols(
        f"{base_func_name}0:{order - 1}",
        cls=sympy.Function,
    )
    substitution_funcs = [dep_func] + [f(dependent_symb) for f in substitution_funcs]
    for i in range(0, order - 1):
        substitute = substitution_funcs[i]
        conv = substitution_funcs[i + 1]
        func_list.append(substitute)
        eq = eq.subs(substitute.diff(dependent_symb), conv)
    func_list.append(conv)
    eq = sympy.Eq(eq.lhs - eq.rhs, 0)
    return eq, func_list


def _convert_diffeq_to_matrix(eq, func_list):
    """Build the state matrix for reduced higher-order ODE systems."""
    func_list.append(eq.atoms(sympy.Derivative).pop())
    a = np.empty((len(func_list) - 1, len(func_list) - 1))
    row = 0
    for item in func_list[1:]:
        coeffless_func = item
        coeff = eq.lhs.coeff(item)
        item = coeffless_func * coeff
        dummy = sympy.Eq(-eq.rhs + item, -eq.lhs + item)
        if coeffless_func == func_list[-1]:
            a[row] = np.array(
                [
                    dummy.rhs.coeff(symb) / coeff
                    for symb in func_list
                    if coeffless_func != symb
                ]
            )
        else:
            a[row] = np.array(
                [1 if z == row + 1 else 0 for z in range(0, len(func_list) - 1)]
            )
        row += 1
    return a


class Differential_Equation:
    """Parsed differential equation container for the UI layer."""

    def __init__(self, strConstants: dict[str, float] | None, Tex: str):
        constants = strConstants or {}
        self._latex = Tex

        parsed_tex = _process_raw_text(Tex)
        expr = parse_latex(parsed_tex, strict=False)
        expr = _replace_derivative(expr)

        derivatives = expr.atoms(Derivative)
        if not derivatives:
            raise ValueError("No derivative was detected in the provided equation.")

        dep_funcs = {deriv.args[0] for deriv in derivatives}
        dep_node = list(dep_funcs)[0]
        deriv = list(derivatives)[0]

        if isinstance(dep_node, AppliedUndef):
            dep_func_name = dep_node.func.__name__
            indep_var = dep_node.args[0]
            dep_func = dep_node
        else:
            dep_func_name = dep_node.name
            v = deriv.args[1]
            indep_var = v[0] if type(v).__name__ in ["tuple", "Tuple"] else v
            dep_func = Function(dep_func_name)(indep_var)

        match_pattern = lambda node: isinstance(node, AppliedUndef) and node != dep_func
        while True:
            new_expr = expr.replace(match_pattern, _fix_implicit_multiplication)
            if new_expr == expr:
                break
            expr = new_expr

        expr = expr.replace(Symbol(dep_func_name), dep_func)

        parse_constants = {
            name: Symbol(name, constant=True, real=True)
            for name in constants.keys()
        }

        self._constant_symbols = parse_constants
        self._constants = list(parse_constants.values())
        self._independent_var = indep_var
        self._dep_func = dep_func
        self._expr = expr.subs(parse_constants)

    @property
    def constants(self) -> list[sympy.Symbol]:
        return self._constants

    @property
    def latex(self) -> str:
        return self._latex

    @property
    def expr(self) -> sympy.Eq:
        return self._expr

    @property
    def dep_func(self) -> sympy.Function:
        return self._dep_func

    @property
    def indep_var(self) -> sympy.Symbol:
        return self._independent_var

    @property
    def order(self) -> int:
        return sympy.ode_order(self.expr, self.dep_func)


class Differential_Equation_Solution:
    """Compute numerical solution for a Differential_Equation object."""

    def __init__(
        self,
        diffeq: Differential_Equation,
        upperRange: int,
        lowerRange: int,
        stepSize: float,
        Y0: list[float],
        solver: str,
        constantValues: dict[str, float] | None = None,
    ):
        self._diffeq = diffeq
        self._range = (upperRange, lowerRange)
        self._step_size = stepSize
        self._Y0 = Y0
        self._solver = solver

        constants = constantValues or {}
        constant_pass = [
            (diffeq._constant_symbols[name], constants[name])
            for name in constants.keys()
            if name in diffeq._constant_symbols
        ]

        expr = diffeq.expr
        dep_func = diffeq.dep_func

        de_sols = {}
        if self._solver == "Base":
            de_sols = solve_ode(
                expr,
                dep_func,
                solver="trapezoidal",
                y0=Y0[0],
                t_span=(lowerRange, upperRange),
                constants=constant_pass,
                step_size=stepSize,
            )
        elif self._solver == "leapfrog":
            if sympy.ode_order(expr, dep_func) == 1:
                de_sols = solve_ode(
                    expr,
                    dep_func,
                    solver="leapfrog",
                    y0=Y0[0],
                    t_span=(lowerRange, upperRange),
                    constants=constant_pass,
                    step_size=stepSize,
                )
            else:
                de_sols = solve_ode(
                    expr,
                    dep_func,
                    solver="leapfrog",
                    y0=Y0[0],
                    v0=Y0[1],
                    t_span=(lowerRange, upperRange),
                    constants=constant_pass,
                    step_size=stepSize,
                )
                de_sols["y"] = de_sols["x"][:, 0]
        elif solver == "RK4":
            if sympy.ode_order(expr, dep_func) == 1:
                f = sympy.lambdify((diffeq.indep_var, dep_func), expr.rhs, modules="numpy")
            else:
                mod_expr, func_list = _solve_higher_order_diffeq(expr, dep_func)
                higher_order_matrix = _convert_diffeq_to_matrix(mod_expr, func_list)

                def f(t, y):
                    return higher_order_matrix @ y

            x0 = np.array([float(value) for value in Y0])
            de_sols = RK4(f, x0=x0, t_span=(lowerRange, upperRange), step_size=stepSize)
            if "v" in de_sols:
                de_sols["y"] = de_sols["x"][:, 0]

        self._de_sols = de_sols

    @property
    def de_sols(self):
        return self._de_sols
