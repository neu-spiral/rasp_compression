"""Accuracy surrogate models A_k(η) for compression rate selection.

Each model maps a per-link compression rate vector η ∈ [0,1]^n_links to an
expected task accuracy scalar.  Models are trained during the ``accuracy_model``
sub-experiment phase using a simulation-based sweep and consumed by optimizers
in the slot loop.  For the LLM backend, training labels are computed on a
**held-out** MMLU / ShareGPT subset (``grad_model_data_seed``), not the same
rows as the fast evaluator used during the run (``eval_seed``).

Available backends
------------------
- ``SurrogateAccuracyModel`` — sklearn regression with optional polynomial
  feature expansion.  Supports ``"linear_monotonic"``, ``"poly2"``,
  ``"poly3"``, ``"gbm"``, ``"rf"``, ``"mlp"``, ``"mlp_small"``.
- ``ConstantAccuracyModel``  — returns a fixed accuracy regardless of η.
  Used as a placeholder when a real model is not yet available.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_BUILDERS: dict[str, Any] = {}


def _get_model_builders() -> dict[str, Any]:
    """Lazily build the sklearn model registry to avoid import cost at module load."""
    global _MODEL_BUILDERS
    if _MODEL_BUILDERS:
        return _MODEL_BUILDERS
    from sklearn.ensemble import (  # noqa: PLC0415
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, Ridge  # noqa: PLC0415
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415

    _MODEL_BUILDERS = {
        "linear_monotonic": lambda: LinearRegression(positive=True),
        "poly2": lambda: Ridge(alpha=0.1),
        "poly3": lambda: Ridge(alpha=0.1),
        "gbm": lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42
        ),
        "rf": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=3000,
            random_state=42,
            early_stopping=True,
        ),
        "mlp_small": lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            random_state=42,
            early_stopping=True,
        ),
    }
    return _MODEL_BUILDERS


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AccuracyModel(ABC):
    """Abstract accuracy surrogate model A_k(η).

    Subclasses implement ``fit`` and ``predict`` for a specific backend.
    ``gradient`` has a default central finite-difference implementation that
    works for any backend at the cost of 2 × n_links predict calls.
    """

    @abstractmethod
    def predict(self, eta: np.ndarray) -> float:
        """Return predicted accuracy for the given per-link compression rate vector.

        Args:
            eta: Compression rate vector of shape ``(n_links,)`` with values
                in ``[0, 1]``.  1.0 = no compression on that link.

        Returns:
            Predicted accuracy in ``[0, 1]``.
        """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: B027
        """Fit the model from training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_links)``, each row a
                per-link η vector.
            y: Accuracy targets of shape ``(n_samples,)``.
        """

    def gradient(self, eta: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Return ∂A_k/∂η_i for each link using central finite differences.

        Args:
            eta: Compression rate vector of shape ``(n_links,)``.
            eps: Finite-difference step size.

        Returns:
            Gradient vector of shape ``(n_links,)``.
        """
        grad = np.zeros_like(eta, dtype=float)
        for i in range(len(eta)):
            e = np.zeros_like(eta)
            e[i] = eps
            grad[i] = (self.predict(eta + e) - self.predict(eta - e)) / (2.0 * eps)
        return grad

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short name of the model backend.

        Returns:
            Model type string for metric tagging.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted or loaded.

        Returns:
            True when predictions are meaningful.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to a pickle file.

        Args:
            path: Destination file path.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AccuracyModel:
        """Restore a previously persisted model.

        Args:
            path: Path to the pickle file written by ``save``.

        Returns:
            Loaded model instance.
        """


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ConstantAccuracyModel(AccuracyModel):
    """Returns a fixed accuracy value regardless of η.

    Useful as a placeholder when no fitted model is available or for
    pipelines where accuracy does not depend on compression rate.

    Args:
        value: Constant accuracy to return (default 1.0).
    """

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def predict(self, eta: np.ndarray) -> float:  # noqa: ARG002
        """Return the fixed accuracy.

        Args:
            eta: Ignored.

        Returns:
            The constant accuracy value.
        """
        return self._value

    @property
    def model_type(self) -> str:
        """Short name of the model backend.

        Returns:
            ``"constant"``
        """
        return "constant"

    @property
    def is_fitted(self) -> bool:
        """Return True (no fitting required).

        Returns:
            Always True.
        """
        return True

    def save(self, path: Path) -> None:
        """Persist to pickle.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model_type": "constant", "value": self._value}, f)

    @classmethod
    def load(cls, path: Path) -> ConstantAccuracyModel:
        """Restore from pickle.

        Args:
            path: Path to pickle file.

        Returns:
            Loaded ConstantAccuracyModel.
        """
        with open(path, "rb") as f:
            d = pickle.load(f)  # noqa: S301
        return cls(value=d.get("value", 1.0))


class SurrogateAccuracyModel(AccuracyModel):
    """Sklearn-backed multivariate regression surrogate A_k(η).

    Accepts a per-link η vector and predicts task accuracy.  For polynomial
    backends (``"poly2"``, ``"poly3"``) a ``PolynomialFeatures`` expansion
    captures cross-link interaction terms.  All backends apply
    ``StandardScaler`` normalisation before fitting and predicting.

    Training splits data 80/20 to evaluate fit quality, then refits on the
    full dataset for deployment — same pattern as the reference estimator.

    Args:
        model_type: One of ``"linear_monotonic"``, ``"poly2"``, ``"poly3"``,
            ``"gbm"``, ``"rf"``, ``"mlp"``, ``"mlp_small"``.
    """

    def __init__(self, model_type: str = "gbm") -> None:
        if model_type not in _get_model_builders():
            raise ValueError(
                f"Unknown model_type {model_type!r}. "
                f"Valid options: {sorted(_get_model_builders())}"
            )
        self._model_type = model_type
        self._model: Any = None
        self._scaler: Any = None
        self._poly: Any = None
        self._n_features: int | None = None
        self._metrics: dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate model from (η_vector, accuracy) training data.

        Applies StandardScaler, optional PolynomialFeatures expansion, then
        trains the backend estimator.  Evaluates on a held-out 20% split and
        logs RMSE/R², then refits on the full dataset.

        Args:
            X: Feature matrix ``(n_samples, n_links)``.
            y: Accuracy targets ``(n_samples,)``.
        """
        from sklearn.metrics import mean_squared_error, r2_score  # noqa: PLC0415
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
        from sklearn.preprocessing import (  # noqa: PLC0415
            PolynomialFeatures,
            StandardScaler,
        )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features = X.shape[1]
        self._scaler = StandardScaler()

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        X_tr_s = self._scaler.fit_transform(X_tr)
        X_te_s = self._scaler.transform(X_te)

        if self._model_type.startswith("poly"):
            degree = int(self._model_type[-1])
            self._poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_tr_s = self._poly.fit_transform(X_tr_s)
            X_te_s = self._poly.transform(X_te_s)

        self._model = _get_model_builders()[self._model_type]()
        self._model.fit(X_tr_s, y_tr)

        y_pred = np.clip(self._model.predict(X_te_s), 0, 1)
        self._metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "test_r2": float(r2_score(y_te, y_pred)),
            "n_train": len(y_tr),
            "n_test": len(y_te),
        }
        logger.info(
            "SurrogateAccuracyModel fitted: type=%s n_links=%d n_train=%d "
            "test_rmse=%.4f test_r2=%.4f",
            self._model_type,
            self._n_features,
            len(y_tr),
            self._metrics["test_rmse"],
            self._metrics["test_r2"],
        )

        # Refit on full dataset for deployment.
        X_all_s = self._scaler.fit_transform(X)
        if self._poly is not None:
            X_all_s = self._poly.fit_transform(X_all_s)
        self._model.fit(X_all_s, y)

    def predict(self, eta: np.ndarray) -> float:
        """Evaluate the fitted model at the given per-link η vector.

        Args:
            eta: Compression rate vector of shape ``(n_links,)``.

        Returns:
            Predicted accuracy clipped to ``[0, 1]``.  Returns ``0.0`` if the
            model has not been fitted.
        """
        if self._model is None or self._scaler is None:
            return 0.0
        x = np.asarray(eta, dtype=float).reshape(1, -1)
        x = self._scaler.transform(x)
        if self._poly is not None:
            x = self._poly.transform(x)
        return float(np.clip(self._model.predict(x)[0], 0.0, 1.0))

    def save(self, path: Path) -> None:
        """Persist the fitted model to a pickle file.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self._model_type,
            "n_features": self._n_features,
            "scaler": self._scaler,
            "poly": self._poly,
            "model": self._model,
            "metrics": self._metrics,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.debug("Saved SurrogateAccuracyModel → %s", path)

    @classmethod
    def load(cls, path: Path) -> SurrogateAccuracyModel:
        """Restore a previously saved model from pickle.

        Args:
            path: Path to the pickle file written by ``save``.

        Returns:
            Loaded SurrogateAccuracyModel instance.
        """
        with open(path, "rb") as f:
            d = pickle.load(f)  # noqa: S301
        m = cls.__new__(cls)
        m._model_type = d["model_type"]
        m._n_features = d["n_features"]
        m._scaler = d["scaler"]
        m._poly = d.get("poly")
        m._model = d["model"]
        m._metrics = d.get("metrics", {})
        logger.debug(
            "Loaded SurrogateAccuracyModel ← %s (type=%s n_links=%s)",
            path,
            m._model_type,
            m._n_features,
        )
        return m

    @property
    def model_type(self) -> str:
        """Short name of the model backend.

        Returns:
            The model_type string, e.g. ``"gbm"``.
        """
        return self._model_type

    @property
    def is_fitted(self) -> bool:
        """Return True when the model has been fitted.

        Returns:
            True after ``fit()`` or ``load()`` with a valid model object.
        """
        return self._model is not None

    @property
    def fit_metrics(self) -> dict[str, float]:
        """Evaluation metrics from the last fit call.

        Returns:
            Dict with ``test_rmse``, ``test_r2``, ``n_train``, ``n_test``.
        """
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_accuracy_model(model_type: str) -> AccuracyModel:
    """Instantiate an accuracy model from a type string.

    Args:
        model_type: One of ``"linear_monotonic"``, ``"poly2"``, ``"poly3"``,
            ``"gbm"``, ``"rf"``, ``"mlp"``, ``"mlp_small"``.
            Legacy ``"poly1"``–``"poly5"`` strings are mapped to the closest
            sklearn poly backend.  Unrecognised strings fall back to
            ``ConstantAccuracyModel``.

    Returns:
        Unfitted accuracy model instance.
    """
    # Legacy poly1–poly5 alias mapping.
    _legacy_poly = {f"poly{d}": f"poly{min(d, 3)}" for d in range(1, 6)}
    resolved = _legacy_poly.get(model_type, model_type)

    if resolved in _get_model_builders():
        return SurrogateAccuracyModel(resolved)

    logger.warning(
        "Unknown accuracy model type %r; falling back to ConstantAccuracyModel",
        model_type,
    )
    return ConstantAccuracyModel()


def load_accuracy_model(path: Path) -> AccuracyModel:
    """Load any persisted accuracy model from a pickle file.

    Inspects the ``model_type`` key in the pickle dict to dispatch to the
    correct class.

    Args:
        path: Path to the pickle file.

    Returns:
        Loaded AccuracyModel instance.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)  # noqa: S301
    model_type = d.get("model_type", "")
    if model_type == "constant":
        return ConstantAccuracyModel.load(path)
    return SurrogateAccuracyModel.load(path)


# ---------------------------------------------------------------------------
# InferenceTask callable factory
# ---------------------------------------------------------------------------


def make_surrogate_callables(
    model: AccuracyModel,
    n_links: int,
) -> tuple[Any, Any, Any]:
    """Wrap an AccuracyModel into the callable triple expected by InferenceTask.

    The model's ``predict`` and ``gradient`` methods are used directly — no
    mean aggregation is applied since the model is vector-input.

    ``accuracy_callable_true`` is identical to ``accuracy_callable`` because
    the surrogate has no "full dataset" mode; it is the model's best estimate
    regardless of evaluation budget.

    Args:
        model: Fitted AccuracyModel instance.
        n_links: Number of inter-node links in the pipeline (used only to
            build the zero gradient fallback for unfitted models).

    Returns:
        Tuple ``(accuracy_callable, accuracy_callable_true, gradient_callable)``
        matching the ``InferenceTask`` constructor signatures.
    """

    def accuracy_callable(eta_np: np.ndarray) -> float:
        return model.predict(np.asarray(eta_np, dtype=float))

    def accuracy_callable_true(eta_np: np.ndarray) -> float:
        return model.predict(np.asarray(eta_np, dtype=float))

    def gradient_callable(eta_np: np.ndarray) -> np.ndarray:
        if not model.is_fitted:
            return np.zeros(n_links, dtype=float)
        return model.gradient(np.asarray(eta_np, dtype=float))

    return accuracy_callable, accuracy_callable_true, gradient_callable