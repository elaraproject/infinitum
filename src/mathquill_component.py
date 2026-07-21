from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "mathquill_component_frontend"
_mathquill_component = components.declare_component(
    "mathquill_input",
    path=str(_COMPONENT_DIR),
)


def mathquill_input(
    label: str,
    value: str = "",
    key: str | None = None,
    placeholder: str = "",
    debounce_ms: int = 180,
    height: int = 140,
) -> str:
    """Render a MathQuill-based Streamlit component and return the LaTeX string."""
    result = _mathquill_component(
        label=label,
        value=value,
        placeholder=placeholder,
        debounceMs=int(debounce_ms),
        height=int(height),
        key=key,
        default=value,
    )
    if result is None:
        return value
    return str(result)
