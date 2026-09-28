"""Stage-2 experiment parameter validation (plan §4.2). Messages mention only public ranges."""
from __future__ import annotations

import math
from typing import Any

from ..contracts import (CategoricalParam, ContinuousParam, IntegerParam, InvalidParameters, PublicTask,
                         ValidParameters)


def validate_parameters(task: PublicTask, raw: Any) -> ValidParameters | InvalidParameters:
    if not isinstance(raw, dict):
        return InvalidParameters(reason="parameters must be an object mapping parameter name to value")
    specs = {p.name: p for p in task.parameters}
    unknown = sorted(set(raw) - set(specs))
    if unknown:
        return InvalidParameters(reason=f"unknown parameter(s): {', '.join(unknown)}")
    missing = sorted(set(specs) - set(raw))
    if missing:
        return InvalidParameters(reason=f"missing required parameter(s): {', '.join(missing)}")
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        v = raw[name]
        if isinstance(spec, CategoricalParam):
            if not isinstance(v, str) or v not in spec.choices:
                return InvalidParameters(reason=f"{name} must be one of {spec.choices}")
            out[name] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return InvalidParameters(reason=f"{name} must be a finite number")
        if isinstance(spec, IntegerParam):
            if float(v) != int(v):
                return InvalidParameters(reason=f"{name} must be an integer")
            v = int(v)
        else:
            v = float(v)
        if not (spec.min <= v <= spec.max):
            unit = f" {spec.unit}" if spec.unit else ""
            return InvalidParameters(reason=f"{name}={v}{unit} is outside the allowed range [{spec.min}, {spec.max}]{unit}")
        out[name] = v
    for c in task.constraints:
        if any(not isinstance(out.get(k), (int, float)) for k in c.coefficients):
            return InvalidParameters(reason=f"constraint references non-numeric parameter: {c.description or c.coefficients}")
        lhs = sum(coef * out[k] for k, coef in c.coefficients.items())
        ok = {"<=": lhs <= c.rhs + c.tolerance, ">=": lhs >= c.rhs - c.tolerance,
              "==": abs(lhs - c.rhs) <= c.tolerance}[c.op]
        if not ok:
            return InvalidParameters(reason=f"constraint violated: {c.description or c.coefficients} {c.op} {c.rhs}")
    return ValidParameters(parameters=out)
