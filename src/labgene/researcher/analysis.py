"""Observation-only analysis tools for the researcher (spec §3.1.2, plan B20).

The tools are built from ONE ResearcherView: its PublicTask and the observations in its history. There is no
file, network, simulator, memory or other-condition path in this module by construction; arguments are strictly
validated against public task names, so path/URL-like or unknown arguments are rejected. Outputs are
AnalysisResult (never Observation); predictions are kind="agent_prediction" and never count as success.
"""
from __future__ import annotations

import math
import re
from typing import Any

from ..contracts import AnalysisResult, CategoricalParam, PublicTask, ResearcherView, canonical_json, sha256_text
from ..providers.base import ToolSpec

TOOL_NAMES = ("describe", "fit", "predict", "nearest")
MAX_POINTS = 20
MAX_K = 10
_PATHLIKE = re.compile(r"://|[\\/]|\.\.|^~|^[A-Za-z]:")
_ONLY = "tools read only this episode's observations and the public task"


class AnalysisError(ValueError):
    """Rejected tool call. Messages mention only public task names and the offending argument."""


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise AnalysisError(f"{what} must be a finite number")
    return float(v)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; raises on (near-)singular systems."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    tol = 1e-9 * max(1.0, max(abs(m[i][i]) for i in range(n)))
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < tol:
            raise AnalysisError("fit is under-determined: observations do not vary enough in the chosen features")
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


class AnalysisTools:
    def __init__(self, view: ResearcherView):
        self.task: PublicTask = view.task
        self.specs_by_name = {p.name: p for p in self.task.parameters}
        self.numeric = [p.name for p in self.task.parameters if not isinstance(p, CategoricalParam)]
        self.categorical = {p.name: p.choices for p in self.task.parameters if isinstance(p, CategoricalParam)}
        self.metrics = {m.name: m.direction for m in self.task.metrics}
        self.obs: list[tuple[str, dict[str, Any], dict[str, float]]] = []
        for h in view.history:
            if h.kind == "observation":
                p = h.payload
                results = {k: float(v) for k, v in (p.get("results") or {}).items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
                self.obs.append((str(p.get("observation_id") or h.action_id), dict(p.get("parameters") or {}), results))

    # ------------------------------------------------------------ model-facing schema

    def specs(self) -> list[ToolSpec]:
        metric = {"type": "string", "enum": sorted(self.metrics)}
        features = {"type": "array", "items": {"type": "string", "enum": self.numeric}}
        degree = {"type": "integer", "enum": [1, 2]}
        where = {"where": {"type": "object", "description": "optional filter: categorical parameter -> choice",
                           "properties": {c: {"type": "string", "enum": ch} for c, ch in self.categorical.items()}}
                 } if self.categorical else {}
        num_point = {"type": "object", "properties": {n: {"type": "number"} for n in self.numeric}}
        any_point = {"type": "object", "properties": {**num_point["properties"],
                                                      **{c: {"type": "string", "enum": ch} for c, ch in self.categorical.items()}}}

        def obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
            return {"type": "object", "properties": props, "required": required}
        return [
            ToolSpec(name="describe", description="Per-metric statistics (n, mean, std, min, max, best) of this "
                     "episode's observations.", parameters=obj({"metrics": {"type": "array", "items": metric}, **where}, [])),
            ToolSpec(name="fit", description="Least-squares linear (degree 1) or quadratic (degree 2, with interactions) "
                     "fit of one metric on numeric parameters, using this episode's observations only.",
                     parameters=obj({"metric": metric, "features": features, "degree": degree, **where}, ["metric"])),
            ToolSpec(name="predict", description="Fit as in `fit`, then predict at given points. Output is an "
                     "agent_prediction (not an observation); flags extrapolation and out-of-range points.",
                     parameters=obj({"metric": metric, "features": features, "degree": degree, **where,
                                     "points": {"type": "array", "items": num_point}}, ["metric", "points"])),
            ToolSpec(name="nearest", description="The k observed points closest to a given point (range-normalised).",
                     parameters=obj({"point": any_point, "k": {"type": "integer"}}, ["point"])),
        ]

    # ------------------------------------------------------------ dispatch + validation

    def run(self, name: str, args: Any) -> AnalysisResult:
        if name not in TOOL_NAMES:
            raise AnalysisError(f"unknown analysis tool {name!r}; available: {', '.join(TOOL_NAMES)}")
        if not isinstance(args, dict):
            raise AnalysisError(f"{name}: arguments must be an object")
        try:
            return getattr(self, f"_{name}")(args)
        except ArithmeticError:            # e.g. 1e308 or a 400-digit integer overflowing float arithmetic
            raise AnalysisError(f"{name}: numeric overflow; argument values are too large") from None

    @staticmethod
    def _keys(tool: str, args: dict[str, Any], allowed: set[str], required: set[str] = frozenset()) -> None:
        extra = sorted(set(args) - allowed)
        if extra:
            raise AnalysisError(f"{tool}: unsupported argument(s) {extra}; {_ONLY}")
        missing = sorted(required - set(args))
        if missing:
            raise AnalysisError(f"{tool}: missing argument(s) {missing}")

    @staticmethod
    def _name(v: Any, allowed: Any, what: str) -> str:
        if not isinstance(v, str) or v not in allowed:
            hint = f" (file paths, URLs and external resources are not accessible; {_ONLY})" \
                if isinstance(v, str) and _PATHLIKE.search(v) else ""
            raise AnalysisError(f"{what} must be one of {sorted(allowed)}{hint}")
        return v

    def _rows(self, args: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, float]]]:
        where = args.get("where", {})
        if not isinstance(where, dict):
            raise AnalysisError("where must be an object mapping categorical parameter -> choice")
        for k, v in where.items():
            self._name(v, self.categorical[self._name(k, self.categorical, "where key")], f"where[{k}]")
        rows = [o for o in self.obs if all(o[1].get(k) == v for k, v in where.items())]
        if not rows:
            raise AnalysisError("no observations in this episode match (run experiments first)")
        return rows

    def _result(self, tool: str, args: dict[str, Any], ids: list[str], method: str, result: dict[str, Any],
                kind: str, uncertainty: dict[str, Any] | None = None) -> AnalysisResult:
        aid = "an-" + sha256_text(canonical_json([tool, args, ids]))[:16]
        return AnalysisResult(analysis_id=aid, input_observation_ids=ids, method=method, result=result,
                              uncertainty=uncertainty, kind=kind)

    # ------------------------------------------------------------ tools

    def _describe(self, args: dict[str, Any]) -> AnalysisResult:
        self._keys("describe", args, {"metrics", "where"})
        metrics = args.get("metrics", sorted(self.metrics))
        if not isinstance(metrics, list) or not metrics:
            raise AnalysisError("metrics must be a non-empty list")
        rows = self._rows(args)
        out = {}
        for m in (self._name(x, self.metrics, "metric") for x in metrics):
            vals = [(oid, r[m]) for oid, _, r in rows if m in r]
            if not vals:
                out[m] = {"n": 0}
                continue
            xs = [v for _, v in vals]
            mean = sum(xs) / len(xs)
            std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
            best = (max if self.metrics[m] == "maximize" else min)(vals, key=lambda t: t[1])
            out[m] = {"n": len(xs), "mean": mean, "std": std, "min": min(xs), "max": max(xs),
                      "best": {"observation_id": best[0], "value": best[1]}, "direction": self.metrics[m]}
        return self._result("describe", args, [o[0] for o in rows], "descriptive_statistics", out, "descriptive_statistic")

    def _fit_model(self, tool: str, args: dict[str, Any]):
        metric = self._name(args.get("metric"), self.metrics, "metric")
        features = args.get("features", self.numeric)
        if not isinstance(features, list) or not features or len(set(map(str, features))) != len(features):
            raise AnalysisError("features must be a non-empty list of distinct numeric parameter names")
        features = [self._name(f, self.numeric, "feature") for f in features]
        degree = args.get("degree", 1)
        if degree not in (1, 2) or isinstance(degree, bool):
            raise AnalysisError("degree must be 1 (linear) or 2 (quadratic)")
        degree = int(degree)
        scale = {}
        for f in features:
            s = self.specs_by_name[f]
            scale[f] = ((s.min + s.max) / 2.0, ((s.max - s.min) / 2.0) or 1.0)
        terms = ["1"] + features
        if degree == 2:
            terms += [f"{a}^2" if a == b else f"{a}*{b}" for i, a in enumerate(features) for b in features[i:]]

        def design(p: dict[str, float]) -> list[float]:
            z = [(p[f] - scale[f][0]) / scale[f][1] for f in features]
            row = [1.0] + z
            if degree == 2:
                row += [z[i] * z[j] for i in range(len(z)) for j in range(i, len(z))]
            return row
        rows = [o for o in self._rows(args) if metric in o[2]
                and all(isinstance(o[1].get(f), (int, float)) and not isinstance(o[1].get(f), bool) for f in features)]
        if len(rows) < len(terms):
            raise AnalysisError(f"{tool}: {len(terms)} terms need at least {len(terms)} observations with {metric}; "
                                f"have {len(rows)}")
        X = [design(o[1]) for o in rows]
        y = [o[2][metric] for o in rows]
        p = len(terms)
        beta = _solve([[sum(r[i] * r[j] for r in X) for j in range(p)] for i in range(p)],
                      [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(p)])
        resid = [yy - sum(b * x for b, x in zip(beta, r)) for r, yy in zip(X, y)]
        ss_res = sum(e * e for e in resid)
        mean = sum(y) / len(y)
        ss_tot = sum((yy - mean) ** 2 for yy in y)
        dof = len(y) - p
        stats = {"n": len(y), "dof": dof, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
                 "residual_std": math.sqrt(ss_res / dof) if dof > 0 else None}
        observed = {f: (min(o[1][f] for o in rows), max(o[1][f] for o in rows)) for f in features}
        return metric, features, degree, rows, terms, beta, design, scale, stats, observed

    def _fit(self, args: dict[str, Any]) -> AnalysisResult:
        self._keys("fit", args, {"metric", "features", "degree", "where"}, {"metric"})
        metric, features, degree, rows, terms, beta, _, scale, stats, _ = self._fit_model("fit", args)
        result = {"metric": metric, "coefficients": dict(zip(terms, beta)),
                  "scaling": {f: {"center": c, "half_range": h} for f, (c, h) in scale.items()},
                  "note": "coefficients apply to scaled features z = (x - center) / half_range"}
        return self._result("fit", args, [o[0] for o in rows], f"least_squares_degree{degree}", result, "analysis", stats)

    def _predict(self, args: dict[str, Any]) -> AnalysisResult:
        self._keys("predict", args, {"metric", "features", "degree", "where", "points"}, {"metric", "points"})
        pts = args["points"]
        if not isinstance(pts, list) or not 1 <= len(pts) <= MAX_POINTS:
            raise AnalysisError(f"points must be a list of 1..{MAX_POINTS} objects")
        metric, features, degree, rows, _, beta, design, _, stats, observed = self._fit_model("predict", args)
        preds = []
        for i, pt in enumerate(pts):
            if not isinstance(pt, dict) or set(pt) != set(features):
                raise AnalysisError(f"points[{i}] must give exactly the features {features}")
            x = {f: _num(pt[f], f"points[{i}].{f}") for f in features}
            specs = {f: self.specs_by_name[f] for f in features}
            violates = None
            if self.task.constraints and all(set(c.coefficients) <= set(x) for c in self.task.constraints):
                violates = any(not {"<=": lhs <= c.rhs + c.tolerance, ">=": lhs >= c.rhs - c.tolerance,
                                    "==": abs(lhs - c.rhs) <= c.tolerance}[c.op]
                               for c in self.task.constraints
                               for lhs in [sum(k * x[n] for n, k in c.coefficients.items())])
            predicted = sum(b * v for b, v in zip(beta, design(x)))
            if not math.isfinite(predicted):
                raise AnalysisError(f"points[{i}]: prediction overflowed; the point is too far from the data")
            preds.append({"point": x, "predicted": predicted,
                          # ponytail: extrapolation = outside the observed bounding box, not the convex hull
                          "extrapolation": any(not observed[f][0] <= x[f] <= observed[f][1] for f in features),
                          "out_of_range": any(not specs[f].min <= x[f] <= specs[f].max for f in features),
                          "violates_constraints": violates})
        result = {"metric": metric, "predictions": preds,
                  "note": "agent_prediction: model estimate, NOT an observation; success is judged only on real experiments"}
        return self._result("predict", args, [o[0] for o in rows], f"least_squares_degree{degree}", result,
                            "agent_prediction", stats)

    def _nearest(self, args: dict[str, Any]) -> AnalysisResult:
        self._keys("nearest", args, {"point", "k"}, {"point"})
        pt, k = args["point"], args.get("k", 3)
        if not isinstance(pt, dict) or not pt:
            raise AnalysisError("point must be a non-empty object of parameter -> value")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise AnalysisError(f"k must be an integer in 1..{MAX_K}")
        for n, v in pt.items():
            self._name(n, self.specs_by_name, "point parameter")
            if n in self.categorical:
                self._name(v, self.categorical[n], f"point.{n}")
            else:
                _num(v, f"point.{n}")
        if not self.obs:
            raise AnalysisError("no observations in this episode yet (run experiments first)")

        def dist(p: dict[str, Any]) -> float:
            d = 0.0
            for n, v in pt.items():
                s, o = self.specs_by_name[n], p.get(n)
                if n in self.categorical:
                    d += 0.0 if o == v else 1.0
                elif isinstance(o, (int, float)) and not isinstance(o, bool):
                    d += ((o - v) / ((s.max - s.min) or 1.0)) ** 2
                else:
                    d += 1.0
            return math.sqrt(d)
        ranked = sorted(self.obs, key=lambda o: (dist(o[1]), o[0]))[:k]
        result = {"neighbours": [{"observation_id": oid, "distance": dist(p), "parameters": p, "results": r}
                                 for oid, p, r in ranked]}
        return self._result("nearest", args, [o[0] for o in ranked], "range_normalised_euclidean", result, "analysis")
