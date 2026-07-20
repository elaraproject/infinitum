from importlib import util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "differential-equation.py"


def load_module():
    spec = util.spec_from_file_location("differential_equation", MODULE_PATH)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_differential_equation_parsing_and_solution():
    module = load_module()

    equation = module.Differential_Equation({}, r"\frac{dy}{dx} = y(1-y)")
    solution = module.Differential_Equation_Solution(
        equation,
        upperRange=1,
        lowerRange=0,
        stepSize=0.1,
        Y0=[0.5],
        solver="Base",
    )

    assert equation.order == 1
    assert equation.latex == r"\frac{dy}{dx} = y(1-y)"
    assert "y" in solution.de_sols
    assert np.isclose(solution.de_sols["y"][0], 0.5)
