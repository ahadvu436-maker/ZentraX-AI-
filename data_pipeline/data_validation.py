"""Schema and type validation.

A :class:`Schema` is a list of :class:`ColumnSpec` objects (dtype, nullability,
ranges, allowed values, uniqueness, regex).  :class:`DataValidator` checks a
frame against a schema, optionally *coercing* columns to the declared dtype
first, and returns both the coerced frame and a structured
:class:`ValidationReport`.

Supported dtype names: ``int``, ``float``, ``str``, ``bool``, ``datetime``,
``category``.  Integer columns that contain missing values are kept as
``float64`` (values are still checked to be integral).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pipeline_core import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    column_kind,
    to_jsonable,
)

__all__ = [
    "DataValidationError",
    "ColumnSpec",
    "Schema",
    "ValidationIssue",
    "ValidationReport",
    "DataValidator",
    "ValidationStep",
]

logger = logging.getLogger(__name__)

VALID_DTYPES = ("int", "float", "str", "bool", "datetime", "category")


class DataValidationError(PipelineError):
    """Raised when data violates its schema (``report`` has the details)."""

    def __init__(self, report: "ValidationReport") -> None:
        shown = "; ".join(i.message for i in report.errors[:5])
        extra = f" (+{len(report.errors) - 5} more)" if len(report.errors) > 5 else ""
        super().__init__(f"Validation failed with {len(report.errors)} error(s): {shown}{extra}")
        self.report = report


@dataclass
class ColumnSpec:
    """Expectations for one column.

    Attributes:
        name: Column name.
        dtype: One of :data:`VALID_DTYPES`.
        nullable: Whether missing values are allowed.
        required: Whether the column must exist.
        min_value / max_value: Inclusive bounds (numbers or datetimes).
        allowed_values: Closed set of permitted values.
        unique: Non-missing values must be unique.
        regex: ``str`` columns must fully match this pattern.
        max_null_ratio: Optional cap on the share of missing values.
        format: ``strptime`` format used when coercing ``datetime`` columns.
    """

    name: str
    dtype: str = "str"
    nullable: bool = True
    required: bool = True
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[Tuple[Any, ...]] = None
    unique: bool = False
    regex: Optional[str] = None
    max_null_ratio: Optional[float] = None
    format: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"Column '{self.name}': dtype must be one of {VALID_DTYPES}")
        if self.allowed_values is not None:
            self.allowed_values = tuple(self.allowed_values)
        if self.regex is not None:
            re.compile(self.regex)  # fail early on a bad pattern
        if self.max_null_ratio is not None and not 0.0 <= self.max_null_ratio <= 1.0:
            raise ValueError(f"Column '{self.name}': max_null_ratio must be within [0, 1]")


@dataclass
class Schema:
    """A table schema.

    Attributes:
        columns: Column specifications.
        allow_extra_columns: If ``False`` unknown columns are errors.
        min_rows / max_rows: Row-count limits.
    """

    columns: List[ColumnSpec] = field(default_factory=list)
    allow_extra_columns: bool = True
    min_rows: int = 1
    max_rows: Optional[int] = None

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate column specs: {dupes}")

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get(self, name: str) -> Optional[ColumnSpec]:
        return next((c for c in self.columns if c.name == name), None)

    def without(self, *names: str) -> "Schema":
        """Copy of the schema without the given columns (e.g. the target)."""
        drop = set(names)
        return Schema(
            [c for c in self.columns if c.name not in drop],
            self.allow_extra_columns, self.min_rows, self.max_rows,
        )

    @classmethod
    def infer(
        cls, df: pd.DataFrame, *, exclude: Iterable[str] = (), nullable: bool = True
    ) -> "Schema":
        """Build a dtype-only schema from a reference frame."""
        skip = set(exclude)
        specs: List[ColumnSpec] = []
        for col in df.columns:
            if col in skip:
                continue
            series = df[col]
            kind = column_kind(series)
            if kind == "numeric":
                dtype = "int" if pd.api.types.is_integer_dtype(series.dtype) else "float"
            elif kind == "bool":
                dtype = "bool"
            elif kind == "datetime":
                dtype = "datetime"
            elif isinstance(series.dtype, pd.CategoricalDtype):
                dtype = "category"
            else:
                dtype = "str"
            specs.append(ColumnSpec(str(col), dtype, nullable=nullable))
        return cls(specs)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "columns": self.columns,
                "allow_extra_columns": self.allow_extra_columns,
                "min_rows": self.min_rows,
                "max_rows": self.max_rows,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Schema":
        return cls(
            [ColumnSpec(**c) for c in payload.get("columns", [])],
            payload.get("allow_extra_columns", True),
            payload.get("min_rows", 1),
            payload.get("max_rows"),
        )


@dataclass
class ValidationIssue:
    """One finding. ``severity`` is ``error`` or ``warning``."""

    check: str
    message: str
    column: Optional[str] = None
    severity: str = "error"
    count: int = 1


@dataclass
class ValidationReport:
    """Outcome of one validation run."""

    rows: int = 0
    columns: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def add(self, check: str, message: str, column: Optional[str] = None,
            count: int = 1, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(check, message, column, severity, count))

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise DataValidationError(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rows": self.rows,
            "columns": self.columns,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [to_jsonable(i) for i in self.issues],
        }


_TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0"}


def _to_bool(value: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if value in (0, 1) else np.nan
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return np.nan


def _coerce(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    """Convert ``series`` to ``spec.dtype``; unconvertible values become missing."""
    dtype = spec.dtype
    if dtype in ("int", "float"):
        out = pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_bool_dtype(out.dtype):
            out = out.astype("int64")
        out = out.astype("float64")
        if dtype == "int":
            bad = np.isinf(out) | (out.notna() & (out.round() != out))
            out = out.mask(bad)
            if not out.isna().any():
                out = out.astype("int64")
        return out
    if dtype == "bool":
        if pd.api.types.is_bool_dtype(series.dtype):
            return series
        out = series.astype(object).map(_to_bool)
        return out.astype(bool) if not out.isna().any() else out
    if dtype == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return series
        return pd.to_datetime(series, errors="coerce", format=spec.format)
    if dtype == "category":
        return series.astype("category")
    # str
    if pd.api.types.is_string_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
        return series
    return series.astype(object).map(lambda v: v if pd.isna(v) else str(v))


def _dtype_matches(series: pd.Series, dtype: str) -> bool:
    kind_dtype = series.dtype
    if dtype == "int":
        return pd.api.types.is_integer_dtype(kind_dtype) and not pd.api.types.is_bool_dtype(kind_dtype)
    if dtype == "float":
        return pd.api.types.is_float_dtype(kind_dtype)
    if dtype == "bool":
        return pd.api.types.is_bool_dtype(kind_dtype)
    if dtype == "datetime":
        return pd.api.types.is_datetime64_any_dtype(kind_dtype)
    if dtype == "category":
        return isinstance(kind_dtype, pd.CategoricalDtype)
    return pd.api.types.is_string_dtype(series) and not isinstance(kind_dtype, pd.CategoricalDtype)


class DataValidator:
    """Validate (and optionally coerce) a DataFrame against a :class:`Schema`.

    Args:
        schema: The expectations.
        coerce: Convert columns to the declared dtype before checking.
            Values that cannot be converted are reported as ``type`` errors.
    """

    def __init__(self, schema: Schema, *, coerce: bool = True) -> None:
        self.schema = schema
        self.coerce = coerce

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        """Return ``(possibly coerced copy, report)``. The input is never mutated."""
        schema = self.schema
        report = ValidationReport(rows=len(df), columns=df.shape[1])
        out = df.copy()

        if df.columns.duplicated().any():
            dup = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
            report.add("duplicate_columns", f"Duplicate column names: {dup}")
            return out, report
        if len(df) < schema.min_rows:
            report.add("min_rows", f"Expected at least {schema.min_rows} row(s), got {len(df)}")
        if schema.max_rows is not None and len(df) > schema.max_rows:
            report.add("max_rows", f"Expected at most {schema.max_rows} row(s), got {len(df)}")
        if not schema.allow_extra_columns:
            extra = [c for c in df.columns if c not in set(schema.column_names)]
            if extra:
                report.add("extra_columns", f"Unexpected columns: {extra}", count=len(extra))

        for spec in schema.columns:
            if spec.name not in out.columns:
                if spec.required:
                    report.add("missing_column", f"Missing required column '{spec.name}'", spec.name)
                continue
            out[spec.name] = self._check_column(out[spec.name], spec, report)

        if report.passed:
            logger.debug("Validation passed (%d rows, %d warnings)", len(df), len(report.warnings))
        else:
            logger.warning("Validation found %d error(s)", len(report.errors))
        return out, report

    def validate_or_raise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and return the coerced frame; raise :class:`DataValidationError` on errors."""
        out, report = self.validate(df)
        report.raise_if_failed()
        return out

    # -- per-column checks ------------------------------------------------ #
    def _check_column(self, series: pd.Series, spec: ColumnSpec, report: ValidationReport) -> pd.Series:
        name = spec.name
        if self.coerce:
            nulls_before = int(series.isna().sum())
            try:
                series = _coerce(series, spec)
            except (ValueError, TypeError, OverflowError) as exc:
                report.add("type", f"Column '{name}' cannot be converted to {spec.dtype}: {exc}", name)
                return series
            failed = int(series.isna().sum()) - nulls_before
            if failed > 0:
                report.add(
                    "type",
                    f"Column '{name}': {failed} value(s) cannot be converted to {spec.dtype}",
                    name, count=failed,
                )
        elif not _dtype_matches(series, spec.dtype):
            report.add("type", f"Column '{name}' has dtype {series.dtype}, expected {spec.dtype}", name)
            return series

        n = len(series)
        null_count = int(series.isna().sum())
        if null_count and not spec.nullable:
            report.add("nullable", f"Column '{name}' has {null_count} missing value(s) but is not nullable",
                       name, count=null_count)
        if spec.max_null_ratio is not None and n and null_count / n > spec.max_null_ratio:
            report.add("max_null_ratio",
                       f"Column '{name}': {null_count / n:.1%} missing exceeds {spec.max_null_ratio:.1%}",
                       name, count=null_count)

        present = series.dropna()
        if not len(present):
            return series

        if spec.min_value is not None or spec.max_value is not None:
            self._check_range(present, spec, report)
        if spec.allowed_values is not None:
            bad = int((~present.isin(spec.allowed_values)).sum())
            if bad:
                report.add("allowed_values", f"Column '{name}': {bad} value(s) outside the allowed set",
                           name, count=bad)
        if spec.unique:
            dupes = int(present.duplicated().sum())
            if dupes:
                report.add("unique", f"Column '{name}': {dupes} duplicate value(s)", name, count=dupes)
        if spec.regex is not None:
            matches = present.astype(str).str.fullmatch(spec.regex)
            bad = int((~matches).sum())
            if bad:
                report.add("regex", f"Column '{name}': {bad} value(s) do not match /{spec.regex}/",
                           name, count=bad)
        return series

    @staticmethod
    def _check_range(present: pd.Series, spec: ColumnSpec, report: ValidationReport) -> None:
        name = spec.name
        kind = column_kind(present)
        if kind not in ("numeric", "datetime"):
            report.add("range", f"Column '{name}': min/max need a numeric or datetime column",
                       name, severity="warning")
            return
        lower, upper = spec.min_value, spec.max_value
        if kind == "datetime":
            lower = pd.Timestamp(lower) if lower is not None else None
            upper = pd.Timestamp(upper) if upper is not None else None
        if lower is not None:
            bad = int((present < lower).sum())
            if bad:
                report.add("min_value", f"Column '{name}': {bad} value(s) below {spec.min_value}",
                           name, count=bad)
        if upper is not None:
            bad = int((present > upper).sum())
            if bad:
                report.add("max_value", f"Column '{name}': {bad} value(s) above {spec.max_value}",
                           name, count=bad)


class ValidationStep(PipelineStep):
    """Validate (and coerce) ``context.data``.

    Args:
        schema: Expected schema. If omitted and ``infer_schema`` is ``True``
            the schema is inferred from the data in ``training`` mode.
        stage: Label under which the report is stored in
            ``context.metadata["validation"]``.
        coerce: Coerce dtypes before checking.
        fail_on_error: Raise :class:`DataValidationError` on errors; otherwise
            log them and continue with the coerced frame.
        infer_schema: Allow schema inference when ``schema`` is ``None``.
    """

    def __init__(
        self,
        schema: Optional[Schema] = None,
        *,
        stage: str = "validation",
        coerce: bool = True,
        fail_on_error: bool = True,
        infer_schema: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        default_name = "validation" if stage == "validation" else f"validation_{stage}"
        super().__init__(name or default_name, **kwargs)
        self.schema = schema
        self.stage = stage
        self.coerce = coerce
        self.fail_on_error = fail_on_error
        self.infer_schema = infer_schema

    def run(self, context: PipelineContext) -> PipelineContext:
        frame = context.require_data()
        schema = self.schema
        if schema is None:
            if not (self.infer_schema and context.mode == "training"):
                raise PipelineError(
                    f"Step '{self.name}' has no schema (inference needs a fitted schema)"
                )
            schema = Schema.infer(frame)
            logger.info("Inferred schema with %d columns", len(schema.columns))
        validated, report = DataValidator(schema, coerce=self.coerce).validate(frame)
        context.metadata.setdefault("validation", {})[self.stage] = report.to_dict()
        for issue in report.warnings:
            logger.warning("Validation warning: %s", issue.message)
        if not report.passed and self.fail_on_error:
            raise DataValidationError(report)
        context.data = validated
        context.artifacts["schema"] = schema
        return context
