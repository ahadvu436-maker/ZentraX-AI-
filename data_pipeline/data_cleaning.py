"""Data cleaning with a fit / transform contract.

``fit`` learns *everything* data dependent from the training set (columns to
drop, imputation values, outlier bounds, columns that are numeric in disguise)
and stores it in a JSON-serialisable state.  ``transform`` applies that state
deterministically, so the exact same cleaning runs at inference time.

Training vs. inference
----------------------
Row-level operations (dropping duplicates, rows that are mostly empty, rows
missing the target) only run in training mode.  ``transform(df, inference=True)``
**never removes or reorders rows** - a prediction must exist for every input row.
Outliers are *clipped*, not dropped, for the same reason.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pipeline_core import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    column_kind,
    to_jsonable,
)

__all__ = ["CleaningError", "CleaningConfig", "CleaningReport", "DataCleaner", "CleaningStep"]

logger = logging.getLogger(__name__)


class CleaningError(PipelineError):
    """Cleaning could not be fitted or applied."""


_NUMERIC_STRATEGIES = ("median", "mean", "constant", "none")
_CATEGORICAL_STRATEGIES = ("mode", "constant", "none")
_DATETIME_STRATEGIES = ("median", "constant", "none")
_STRATEGIES_BY_KIND = {
    "numeric": _NUMERIC_STRATEGIES,
    "categorical": _CATEGORICAL_STRATEGIES,
    "bool": _CATEGORICAL_STRATEGIES,
    "datetime": _DATETIME_STRATEGIES,
}


@dataclass
class CleaningConfig:
    """Declarative cleaning policy.

    Attributes:
        strip_strings: Trim whitespace in text columns.
        na_tokens: Case-insensitive strings treated as missing.
        infer_numeric: Convert text columns whose non-missing values are all
            numeric (e.g. ``"12"``, ``"?"``) to numbers. Opt out per column
            with ``no_infer_columns`` (e.g. zip codes, IDs with leading zeros).
        drop_duplicates / duplicate_subset: Remove duplicate rows (training only).
        drop_rows_missing: Rows with a missing value in any of these columns
            are removed in training (typically the target).
        max_missing_row_ratio: Training rows with a larger share of missing
            cells are removed. ``1.0`` disables the check.
        max_missing_col_ratio: Columns with a larger share of missing values
            are dropped (never for ``protected_columns``).
        numeric_strategy / categorical_strategy / datetime_strategy: Default
            imputation per column kind; see ``*_fill_value`` for constants.
        column_strategies: Per-column override, e.g.
            ``{"income": {"strategy": "mean"}, "tier": {"strategy": "constant", "value": "none"}}``.
        add_missing_indicators: Add ``<col>__was_missing`` flags for columns
            that had missing values in training.
        outlier_method: ``None``, ``"iqr"`` or ``"zscore"`` (values are clipped).
        outlier_threshold: IQR multiplier (default 1.5) or z-score (default 3.0).
        protected_columns: Never imputed, clipped or dropped (e.g. the target).
    """

    strip_strings: bool = True
    na_tokens: Tuple[str, ...] = ("", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?")
    infer_numeric: bool = True
    no_infer_columns: Tuple[str, ...] = ()
    drop_duplicates: bool = True
    duplicate_subset: Optional[Tuple[str, ...]] = None
    drop_rows_missing: Tuple[str, ...] = ()
    max_missing_row_ratio: float = 0.8
    max_missing_col_ratio: float = 0.6
    numeric_strategy: str = "median"
    numeric_fill_value: float = 0.0
    categorical_strategy: str = "mode"
    categorical_fill_value: str = "missing"
    datetime_strategy: str = "none"
    column_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    add_missing_indicators: bool = False
    outlier_method: Optional[str] = None
    outlier_threshold: Optional[float] = None
    protected_columns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.na_tokens = tuple(str(t).lower() for t in self.na_tokens)
        self.no_infer_columns = tuple(self.no_infer_columns)
        self.drop_rows_missing = tuple(self.drop_rows_missing)
        self.protected_columns = tuple(self.protected_columns)
        if self.duplicate_subset is not None:
            self.duplicate_subset = tuple(self.duplicate_subset)
        if self.numeric_strategy not in _NUMERIC_STRATEGIES:
            raise CleaningError(f"numeric_strategy must be one of {_NUMERIC_STRATEGIES}")
        if self.categorical_strategy not in _CATEGORICAL_STRATEGIES:
            raise CleaningError(f"categorical_strategy must be one of {_CATEGORICAL_STRATEGIES}")
        if self.datetime_strategy not in _DATETIME_STRATEGIES:
            raise CleaningError(f"datetime_strategy must be one of {_DATETIME_STRATEGIES}")
        if self.outlier_method not in (None, "iqr", "zscore"):
            raise CleaningError("outlier_method must be None, 'iqr' or 'zscore'")
        for ratio_name in ("max_missing_row_ratio", "max_missing_col_ratio"):
            if not 0.0 <= getattr(self, ratio_name) <= 1.0:
                raise CleaningError(f"{ratio_name} must be within [0, 1]")


@dataclass
class CleaningReport:
    """What the last ``transform`` call did."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    rows_dropped_missing_required: int = 0
    rows_dropped_too_empty: int = 0
    columns_dropped: List[str] = field(default_factory=list)
    values_imputed: Dict[str, int] = field(default_factory=dict)
    values_clipped: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _CleaningState:
    columns: List[str] = field(default_factory=list)
    inferred_numeric: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    fill_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clip_bounds: Dict[str, List[float]] = field(default_factory=dict)
    indicator_columns: List[str] = field(default_factory=list)


class DataCleaner:
    """Fit/transform data cleaner. See the module docstring for semantics."""

    def __init__(self, config: Optional[CleaningConfig] = None) -> None:
        self.config = config or CleaningConfig()
        self._state: Optional[_CleaningState] = None
        self.last_report: Optional[CleaningReport] = None

    # -- state ------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    def _require_state(self) -> _CleaningState:
        if self._state is None:
            raise CleaningError("DataCleaner is not fitted; call fit() first")
        return self._state

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable config + fitted state."""
        return {
            "config": to_jsonable(self.config),
            "state": to_jsonable(self._state) if self._state else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataCleaner":
        cleaner = cls(CleaningConfig(**payload["config"]))
        if payload.get("state") is not None:
            cleaner._state = _CleaningState(**payload["state"])
        return cleaner

    # -- text / type normalisation ---------------------------------------- #
    def _clean_text(self, series: pd.Series) -> pd.Series:
        cfg = self.config
        tokens = set(cfg.na_tokens)
        if pd.api.types.is_string_dtype(series):
            text = series.str.strip() if cfg.strip_strings else series
            return text.mask(text.str.lower().isin(tokens))

        def fix(value: Any) -> Any:
            if isinstance(value, str):
                trimmed = value.strip() if cfg.strip_strings else value
                return np.nan if trimmed.lower() in tokens else trimmed
            return value

        return series.map(fix)

    def _prepare(
        self, df: pd.DataFrame, inferred: Optional[Sequence[str]]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Normalise missing tokens and numeric-in-disguise columns.

        ``inferred=None`` means *fit*: detect numeric text columns.  Otherwise
        the previously learned list is applied.
        """
        cfg = self.config
        out = df.copy()
        learned: List[str] = []
        for col in list(out.columns):
            series = out[col]
            if column_kind(series) != "categorical" or isinstance(series.dtype, pd.CategoricalDtype):
                continue
            series = self._clean_text(series)
            if inferred is None:
                if cfg.infer_numeric and col not in cfg.no_infer_columns:
                    numeric = pd.to_numeric(series, errors="coerce")
                    non_missing = int(series.notna().sum())
                    if non_missing and int(numeric.notna().sum()) == non_missing:
                        series = numeric
                        learned.append(col)
            elif col in inferred:
                series = pd.to_numeric(series, errors="coerce")
            out[col] = series
        return out, learned

    # -- fit --------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn cleaning parameters from ``df`` (the training data)."""
        cfg = self.config
        if df.empty:
            raise CleaningError("Cannot fit on an empty DataFrame")
        unknown = [
            c for c in (*cfg.protected_columns, *cfg.drop_rows_missing) if c not in df.columns
        ]
        if unknown:
            raise CleaningError(f"Configured columns not found in data: {sorted(set(unknown))}")

        work, inferred = self._prepare(df, inferred=None)
        protected = set(cfg.protected_columns)
        ratios = work.isna().mean()
        dropped = [
            c for c in work.columns if c not in protected and ratios[c] > cfg.max_missing_col_ratio
        ]
        kept = [c for c in work.columns if c not in dropped]

        fill_values: Dict[str, Dict[str, Any]] = {}
        clip_bounds: Dict[str, List[float]] = {}
        indicators: List[str] = []

        for col in kept:
            if col in protected:
                continue
            series = work[col]
            kind = column_kind(series)
            if series.isna().any():
                indicators.append(col)

            strategy, constant = self._resolve_rule(col, kind)
            if strategy != "none":
                value = self._fit_fill_value(series, kind, strategy, constant)
                if value is not None:
                    fill_values[col] = {"kind": kind, "value": value}

            if cfg.outlier_method and kind == "numeric":
                bounds = self._fit_bounds(series)
                if bounds is not None:
                    clip_bounds[col] = bounds

        self._state = _CleaningState(
            columns=list(work.columns),
            inferred_numeric=inferred,
            dropped_columns=dropped,
            fill_values=fill_values,
            clip_bounds=clip_bounds,
            indicator_columns=indicators if cfg.add_missing_indicators else [],
        )
        logger.info(
            "Cleaner fitted: %d columns kept, %d dropped, %d imputers, %d clip bounds",
            len(kept), len(dropped), len(fill_values), len(clip_bounds),
        )
        return self

    def _resolve_rule(self, col: str, kind: str) -> Tuple[str, Any]:
        cfg = self.config
        defaults = {
            "numeric": (cfg.numeric_strategy, cfg.numeric_fill_value),
            "categorical": (cfg.categorical_strategy, cfg.categorical_fill_value),
            "bool": (cfg.categorical_strategy, cfg.categorical_fill_value),
            "datetime": (cfg.datetime_strategy, None),
        }
        strategy, constant = defaults[kind]
        override = cfg.column_strategies.get(col)
        if override:
            strategy = override.get("strategy", strategy)
            constant = override.get("value", constant)
        if strategy not in _STRATEGIES_BY_KIND[kind]:
            raise CleaningError(
                f"Strategy '{strategy}' is invalid for {kind} column '{col}'; "
                f"choose from {_STRATEGIES_BY_KIND[kind]}"
            )
        return strategy, constant

    @staticmethod
    def _is_integral(series: pd.Series) -> bool:
        values = series.dropna()
        return bool(len(values)) and bool(np.all(np.mod(values.to_numpy(dtype="float64"), 1) == 0))

    def _fit_fill_value(self, series: pd.Series, kind: str, strategy: str, constant: Any) -> Any:
        values = series.dropna()
        if kind == "numeric":
            if strategy == "constant":
                return float(constant if constant is not None else 0.0)
            statistic = float(values.median() if strategy == "median" else values.mean()) if len(values) else math.nan
            if not math.isfinite(statistic):
                return float(self.config.numeric_fill_value)
            # Keep integer columns integral so schema checks still hold.
            return float(round(statistic)) if self._is_integral(series) else statistic
        if kind in ("categorical", "bool"):
            if strategy == "mode":
                modes = values.mode()
                if len(modes):
                    mode = modes.iloc[0]
                    return bool(mode) if kind == "bool" else str(mode)
            return constant if kind == "bool" and isinstance(constant, bool) else str(constant)
        # datetime
        if strategy == "constant":
            return pd.Timestamp(constant).isoformat() if constant is not None else None
        if not len(values):
            return None
        ordered = values.sort_values()
        return pd.Timestamp(ordered.iloc[(len(ordered) - 1) // 2]).isoformat()

    def _fit_bounds(self, series: pd.Series) -> Optional[List[float]]:
        cfg = self.config
        values = series.dropna().astype("float64")
        if len(values) < 4:
            return None
        if cfg.outlier_method == "iqr":
            k = cfg.outlier_threshold if cfg.outlier_threshold is not None else 1.5
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            if not spread > 0:
                return None
            lower, upper = q1 - k * spread, q3 + k * spread
        else:
            k = cfg.outlier_threshold if cfg.outlier_threshold is not None else 3.0
            mean, std = float(values.mean()), float(values.std(ddof=0))
            if not std > 0:
                return None
            lower, upper = mean - k * std, mean + k * std
        if self._is_integral(series):
            lower, upper = math.floor(lower), math.ceil(upper)
        return [float(lower), float(upper)]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return the cleaned training frame."""
        return self.fit(df).transform(df, inference=False)

    # -- transform --------------------------------------------------------- #
    def transform(self, df: pd.DataFrame, *, inference: bool = False) -> pd.DataFrame:
        """Apply the fitted cleaning.

        Args:
            df: Raw frame.
            inference: When ``True`` rows are never dropped or reordered.
        """
        state = self._require_state()
        cfg = self.config
        report = CleaningReport(rows_in=len(df))

        required = (
            set(state.fill_values) | set(state.clip_bounds) | set(state.indicator_columns)
        ) - set(cfg.protected_columns)
        missing_cols = sorted(c for c in required if c not in df.columns)
        if missing_cols:
            raise CleaningError(f"Input is missing columns seen during fit: {missing_cols}")

        out, _ = self._prepare(df, inferred=state.inferred_numeric)
        dropped = [c for c in state.dropped_columns if c in out.columns]
        out = out.drop(columns=dropped)
        report.columns_dropped = dropped

        if not inference:
            if cfg.drop_rows_missing:
                before = len(out)
                out = out.dropna(subset=list(cfg.drop_rows_missing))
                report.rows_dropped_missing_required = before - len(out)
            if cfg.drop_duplicates:
                before = len(out)
                subset = list(cfg.duplicate_subset) if cfg.duplicate_subset else None
                out = out.drop_duplicates(subset=subset)
                report.duplicates_removed = before - len(out)
            if cfg.max_missing_row_ratio < 1.0 and out.shape[1]:
                before = len(out)
                out = out[out.isna().mean(axis=1) <= cfg.max_missing_row_ratio]
                report.rows_dropped_too_empty = before - len(out)
            out = out.reset_index(drop=True)

        if state.indicator_columns:
            flags = {
                f"{col}__was_missing": out[col].isna().astype("int8")
                for col in state.indicator_columns
                if col in out.columns
            }
            out = pd.concat([out, pd.DataFrame(flags, index=out.index)], axis=1)

        for col, spec in state.fill_values.items():
            if col not in out.columns:
                continue
            series = out[col]
            n_missing = int(series.isna().sum())
            if not n_missing:
                continue
            value: Any = spec["value"]
            if spec["kind"] == "datetime":
                value = pd.Timestamp(value)
            elif spec["kind"] == "numeric":
                value = float(value)
            if isinstance(series.dtype, pd.CategoricalDtype) and value not in series.cat.categories:
                series = series.cat.add_categories([value])
            out[col] = series.fillna(value)
            report.values_imputed[col] = n_missing

        for col, (lower, upper) in state.clip_bounds.items():
            if col not in out.columns:
                continue
            series = out[col]
            n_out = int(((series < lower) | (series > upper)).sum())
            if n_out:
                out[col] = series.clip(lower=lower, upper=upper)
                report.values_clipped[col] = n_out

        report.rows_out = len(out)
        self.last_report = report
        if inference and report.rows_out != report.rows_in:  # pragma: no cover - invariant
            raise CleaningError("Internal error: rows changed during inference cleaning")
        return out


class CleaningStep(PipelineStep):
    """Fit + clean in ``training`` mode, clean-only in ``inference`` mode.

    Args:
        cleaner: A :class:`DataCleaner` (fitted or not) or ``None``.
        config: Used to build a new cleaner when ``cleaner`` is ``None``.
        refit: In training mode refit even if the cleaner is already fitted.
    """

    def __init__(
        self,
        cleaner: Optional[DataCleaner] = None,
        config: Optional[CleaningConfig] = None,
        *,
        name: str = "cleaning",
        refit: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.cleaner = cleaner or DataCleaner(config)
        self.refit = refit

    def run(self, context: PipelineContext) -> PipelineContext:
        frame = context.require_data()
        if context.mode == "training":
            if self.refit or not self.cleaner.is_fitted:
                self.cleaner.fit(frame)
            cleaned = self.cleaner.transform(frame, inference=False)
        else:
            cleaned = self.cleaner.transform(frame, inference=True)
        context.data = cleaned
        context.artifacts["cleaner"] = self.cleaner
        report = self.cleaner.last_report
        context.metadata["cleaning"] = report.to_dict() if report else {}
        return context
