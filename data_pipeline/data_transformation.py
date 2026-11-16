"""Feature transformation with a fit / transform contract.

:class:`FeatureTransformer` turns a cleaned frame into a purely numeric
feature matrix:

* numeric columns   -> optional signed-log, then standard / min-max / robust scaling
* boolean columns   -> 0.0 / 1.0
* datetime columns  -> calendar parts (year, month, dayofweek ...)
* text / category   -> one-hot, ordinal or frequency encoding (high-cardinality
  columns automatically fall back to frequency encoding); rare categories can
  be grouped into ``__other__``

Everything learned in ``fit`` lives in a JSON-serialisable state, so the same
transformation can be re-applied at inference time without pickling.
Columns listed in ``exclude_columns`` (target, ids ...) pass through untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .pipeline_core import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    atomic_write_json,
    columns_of_kind,
    to_jsonable,
)

__all__ = [
    "TransformationError",
    "TransformConfig",
    "FeatureTransformer",
    "TransformationStep",
]

logger = logging.getLogger(__name__)

OTHER_TOKEN = "__other__"
MISSING_TOKEN = "__missing__"
_SCALINGS = ("standard", "minmax", "robust", "none")
_ENCODINGS = ("onehot", "ordinal", "frequency", "drop")
_DATETIME_PARTS = ("year", "month", "day", "dayofweek", "hour", "quarter", "dayofyear", "is_weekend")


class TransformationError(PipelineError):
    """Transformation could not be fitted or applied."""


@dataclass
class TransformConfig:
    """Declarative feature-engineering policy.

    Attributes:
        numeric_scaling: ``standard``, ``minmax``, ``robust`` or ``none``.
        categorical_encoding: ``onehot``, ``ordinal``, ``frequency`` or ``drop``.
        max_onehot_cardinality: One-hot columns with more distinct values use
            frequency encoding instead.
        rare_threshold: Categories rarer than this share (0-1) of the training
            rows are merged into ``__other__`` (``0`` disables).
        datetime_parts: Calendar features extracted from datetime columns.
        log_columns: Numeric columns that get ``sign(x) * log1p(|x|)`` first.
        exclude_columns: Pass-through columns (target, ids ...).
        fill_na_value: Value used for any NaN left in the features (``None`` keeps NaN).
    """

    numeric_scaling: str = "standard"
    categorical_encoding: str = "onehot"
    max_onehot_cardinality: int = 20
    rare_threshold: float = 0.0
    datetime_parts: Tuple[str, ...] = ("year", "month", "dayofweek")
    log_columns: Tuple[str, ...] = ()
    exclude_columns: Tuple[str, ...] = ()
    fill_na_value: Optional[float] = 0.0

    def __post_init__(self) -> None:
        self.datetime_parts = tuple(self.datetime_parts)
        self.log_columns = tuple(self.log_columns)
        self.exclude_columns = tuple(self.exclude_columns)
        if self.numeric_scaling not in _SCALINGS:
            raise TransformationError(f"numeric_scaling must be one of {_SCALINGS}")
        if self.categorical_encoding not in _ENCODINGS:
            raise TransformationError(f"categorical_encoding must be one of {_ENCODINGS}")
        bad = [p for p in self.datetime_parts if p not in _DATETIME_PARTS]
        if bad:
            raise TransformationError(f"Unknown datetime parts {bad}; choose from {_DATETIME_PARTS}")
        if not 0.0 <= self.rare_threshold < 1.0:
            raise TransformationError("rare_threshold must be within [0, 1)")
        if self.max_onehot_cardinality < 1:
            raise TransformationError("max_onehot_cardinality must be >= 1")


@dataclass
class _TransformState:
    numeric: Dict[str, Dict[str, float]] = field(default_factory=dict)
    boolean: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    encodings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


def _as_text(series: pd.Series) -> pd.Series:
    """Stringify a column; missing values become ``__missing__``."""
    as_object = series.astype(object)
    return as_object.where(series.notna(), MISSING_TOKEN).astype(str)


class FeatureTransformer:
    """Fit/transform feature engineering. See the module docstring."""

    def __init__(self, config: Optional[TransformConfig] = None) -> None:
        self.config = config or TransformConfig()
        self._state: Optional[_TransformState] = None

    # -- state ------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def feature_names(self) -> List[str]:
        """Names of the generated feature columns (after ``fit``)."""
        return list(self._require_state().feature_names)

    def _require_state(self) -> _TransformState:
        if self._state is None:
            raise TransformationError("FeatureTransformer is not fitted; call fit() first")
        return self._state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": to_jsonable(self.config),
            "state": to_jsonable(self._state) if self._state else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureTransformer":
        transformer = cls(TransformConfig(**payload["config"]))
        if payload.get("state") is not None:
            transformer._state = _TransformState(**payload["state"])
        return transformer

    def save(self, path: Union[str, Path]) -> None:
        """Write the fitted transformer as JSON (atomic)."""
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureTransformer":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- fit --------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        """Learn scalers, category maps and the output column order."""
        cfg = self.config
        if df.empty:
            raise TransformationError("Cannot fit on an empty DataFrame")
        exclude = set(cfg.exclude_columns)
        numeric = columns_of_kind(df, "numeric", exclude)
        booleans = columns_of_kind(df, "bool", exclude)
        datetimes = columns_of_kind(df, "datetime", exclude)
        categoricals = columns_of_kind(df, "categorical", exclude)

        unknown_log = [c for c in cfg.log_columns if c not in numeric]
        if unknown_log:
            raise TransformationError(f"log_columns must be numeric feature columns: {unknown_log}")
        log_set = set(cfg.log_columns)

        state = _TransformState(boolean=booleans, datetime=datetimes)
        names: List[str] = []

        for col in numeric:
            values = self._numeric_series(df[col], col in log_set)
            state.numeric[col] = self._fit_scaler(values, log=col in log_set)
            names.append(col)
        names.extend(booleans)
        for col in datetimes:
            names.extend(f"{col}__{part}" for part in cfg.datetime_parts)

        n_rows = len(df)
        for col in categoricals:
            if cfg.categorical_encoding == "drop":
                logger.warning("Dropping categorical column '%s' (encoding='drop')", col)
                continue
            text = _as_text(df[col])
            counts = text.value_counts()
            encoding = cfg.categorical_encoding
            if encoding == "onehot" and len(counts) > cfg.max_onehot_cardinality:
                logger.info("'%s' has %d categories -> frequency encoding", col, len(counts))
                encoding = "frequency"

            spec: Dict[str, Any] = {"type": encoding}
            keep = counts
            if cfg.rare_threshold > 0:
                keep = counts[counts / n_rows >= cfg.rare_threshold]
                spec["has_other"] = bool(len(keep) < len(counts))
            else:
                spec["has_other"] = False

            if encoding == "frequency":
                spec["frequencies"] = {str(k): float(v / n_rows) for k, v in counts.items()}
                names.append(f"{col}__freq")
            else:
                categories = sorted(str(k) for k in keep.index)
                if spec["has_other"]:
                    categories.append(OTHER_TOKEN)
                if encoding == "onehot":
                    spec["categories"] = categories
                    names.extend(f"{col}={c}" for c in categories)
                else:
                    spec["mapping"] = {c: i for i, c in enumerate(categories)}
                    names.append(col)
            state.encodings[col] = spec

        state.feature_names = names
        self._state = state
        logger.info(
            "Transformer fitted: %d numeric, %d bool, %d datetime, %d categorical -> %d features",
            len(numeric), len(booleans), len(datetimes), len(state.encodings), len(names),
        )
        return self

    @staticmethod
    def _numeric_series(series: pd.Series, log: bool) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").astype("float64")
        if log:
            values = np.sign(values) * np.log1p(np.abs(values))
        return values

    def _fit_scaler(self, values: pd.Series, *, log: bool) -> Dict[str, float]:
        method = self.config.numeric_scaling
        finite = values[np.isfinite(values)]
        center, scale = 0.0, 1.0
        if len(finite) and method != "none":
            if method == "standard":
                center, scale = float(finite.mean()), float(finite.std(ddof=0))
            elif method == "minmax":
                center, scale = float(finite.min()), float(finite.max() - finite.min())
            else:  # robust
                q1, q3 = finite.quantile([0.25, 0.75])
                center, scale = float(finite.median()), float(q3 - q1)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        if not np.isfinite(center):
            center = 0.0
        return {"center": center, "scale": scale, "log": float(log)}

    # -- transform --------------------------------------------------------- #
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``[features..., pass-through columns...]`` with the fitted column order."""
        state = self._require_state()
        cfg = self.config
        needed = [*state.numeric, *state.boolean, *state.datetime, *state.encodings]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise TransformationError(f"Input is missing columns seen during fit: {missing}")

        cols: Dict[str, Any] = {}
        for col, params in state.numeric.items():
            values = self._numeric_series(df[col], bool(params["log"]))
            cols[col] = (values - params["center"]) / params["scale"]
        for col in state.boolean:
            cols[col] = df[col].astype("float64")
        for col in state.datetime:
            parsed = pd.to_datetime(df[col], errors="coerce")
            for part in cfg.datetime_parts:
                if part == "is_weekend":
                    cols[f"{col}__{part}"] = (parsed.dt.dayofweek >= 5).astype("float64").where(parsed.notna())
                else:
                    cols[f"{col}__{part}"] = getattr(parsed.dt, part).astype("float64")

        for col, spec in state.encodings.items():
            text = _as_text(df[col])
            kind = spec["type"]
            if kind == "frequency":
                cols[f"{col}__freq"] = text.map(spec["frequencies"]).fillna(0.0).astype("float64")
                continue
            if kind == "onehot":
                categories = spec["categories"]
            else:
                categories = list(spec["mapping"])
            known = set(categories)
            if spec["has_other"]:
                text = text.where(text.isin(known), OTHER_TOKEN)
            if kind == "onehot":
                for category in categories:
                    cols[f"{col}={category}"] = (text == category).astype("float64")
            else:
                cols[col] = text.map(spec["mapping"]).fillna(-1).astype("float64")

        features = pd.DataFrame(cols, index=df.index).reindex(columns=state.feature_names)
        if cfg.fill_na_value is not None:
            n_na = int(features.isna().to_numpy().sum())
            if n_na:
                logger.info("Filling %d NaN feature cell(s) with %s", n_na, cfg.fill_na_value)
                features = features.fillna(cfg.fill_na_value)
        passthrough = [c for c in cfg.exclude_columns if c in df.columns]
        return pd.concat([features, df[passthrough]], axis=1) if passthrough else features


class TransformationStep(PipelineStep):
    """Fit + transform in ``training`` mode, transform-only in ``inference`` mode."""

    def __init__(
        self,
        transformer: Optional[FeatureTransformer] = None,
        config: Optional[TransformConfig] = None,
        *,
        name: str = "transformation",
        refit: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.transformer = transformer or FeatureTransformer(config)
        self.refit = refit

    def run(self, context: PipelineContext) -> PipelineContext:
        frame = context.require_data()
        if context.mode == "training":
            if self.refit or not self.transformer.is_fitted:
                self.transformer.fit(frame)
        elif not self.transformer.is_fitted:
            raise TransformationError("Inference requires a fitted transformer")
        context.data = self.transformer.transform(frame)
        context.artifacts["transformer"] = self.transformer
        names = self.transformer.feature_names
        context.metadata["transformation"] = {"n_features": len(names), "feature_names": names}
        return context
