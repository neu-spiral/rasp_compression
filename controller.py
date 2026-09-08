"""
controller.py  –  Central controller for CSI-aware single-task deployment
==========================================================================

Runs on any machine (can be one of the Pis or a separate laptop on the same WiFi).

1. Listens for UDP metric reports from all pipeline nodes.
2. Assembles per-slot system state: C_i(t), tau_i, a_i, D_actual(t).
3. Calls the optimizer (uniform stub or CSI-aware closed-form).
4. Broadcasts per-node ControlDirective (eta_i, codec) via UDP to each node.

Usage:
    # Uniform (static eta):
    python controller.py --nodes node_list.txt --port 9999 --algorithm uniform --default_eta 0.7 --method topk

    # CSI-aware (adaptive per-link eta):
    python controller.py --nodes node_list.txt --port 9999 --algorithm csi_aware --R_t 2.0 --method topk --eta_min 0.1
"""

import argparse
import json
import os
import socket
import threading
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CTRL] %(message)s")


# ============================================================================
# Metric Collector — receives UDP reports from pipeline nodes
# ============================================================================

class MetricCollector:
    """
    Listens for UDP metric reports from pipeline nodes.
    Stores the latest report per node_id, and per-step reports for D_actual.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(1.0)

        # Latest metrics per node_id (for tau, c_i)
        self.latest: Dict[int, Dict[str, Any]] = {}
        # Per-step storage: step -> {node_id -> report}
        self.by_step: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        # Full history for logging/analysis
        self.history: List[Dict[str, Any]] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logging.info(f"Metric collector listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._sock.close()

    def _listen_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                node_id = msg["node_id"]
                step = msg.get("step", -1)
                self.latest[node_id] = msg
                self.by_step[step][node_id] = msg
                self.history.append(msg)
            except socket.timeout:
                continue
            except Exception as e:
                logging.warning(f"Metric parse error: {e}")

    def get_system_state(self, num_links: int, link_of_layer: Dict[int, int],
                         first_pipeline_node: int = 1,
                         last_pipeline_node: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Assemble the current system state from latest node reports.

        Returns None if no node has reported yet.

        Parameters
        ----------
        num_links : int
            Number of machine cut-point links in the pipeline.
        link_of_layer : dict
            Maps a cut-point layer id (node_id) -> link index in [0, num_links).
            Only cut-point nodes contribute c_i / a_i to a link; every node still
            contributes its compute time tau (aggregated per machine by the caller).
        first_pipeline_node, last_pipeline_node : int
            First / last pipeline layer ids, used for the end-to-end D_actual.

        Returns dict with:
            c_t:      np.array [num_links] — throughput per cut-point link (bytes/sec)
            c_probed: np.array [num_links] — probed throughput per cut-point link
            tau:      dict node_id -> compute time (all reporting nodes)
            a_i:      np.array [num_links] — uncompressed activation size per link (bytes)
            D_actual: float — end-to-end delay (if a complete step is available)
        """
        if len(self.latest) == 0:
            return None

        c_t = np.zeros(num_links)
        c_probed = np.zeros(num_links)
        a_i = np.zeros(num_links)
        tau = {}
        d_actual = None

        for node_id, report in self.latest.items():
            tau[node_id] = report["tau_i"]

            # Only cut-point nodes carry a compression link.
            link_idx = link_of_layer.get(node_id)
            if link_idx is not None:
                c_i = report.get("c_i", 0.0)
                if c_i > 0:
                    c_t[link_idx] = c_i
                cp = report.get("c_probed_i", 0.0)
                if cp > 0:
                    c_probed[link_idx] = cp
                raw = report.get("raw_bytes_i", 0)
                if raw > 0:
                    a_i[link_idx] = raw

        # Compute D_actual from step-matched reports (first arrival -> last departure)
        if last_pipeline_node is not None:
            for step in sorted(self.by_step.keys(), reverse=True):
                step_reports = self.by_step[step]
                if first_pipeline_node in step_reports and last_pipeline_node in step_reports:
                    t_start = step_reports[first_pipeline_node]["t_arrival_wall"]
                    t_end = step_reports[last_pipeline_node]["t_departure_wall"]
                    d_actual = t_end - t_start
                    break

        # Clean up old steps to prevent memory growth
        completed_steps = sorted(self.by_step.keys())
        if len(completed_steps) > 100:
            for old_step in completed_steps[:-50]:
                del self.by_step[old_step]

        return {
            "c_t": c_t,
            "c_probed": c_probed,
            "tau": tau,
            "a_i": a_i,
            "D_actual": d_actual,
        }


# ============================================================================
# Uniform Optimizer — static eta (baseline / stub)
# ============================================================================

class UniformOptimizer:
    """
    Returns uniform eta across all links. Ignores all measurements.
    """

    def __init__(self, num_links: int, default_eta: float = 1.0,
                 method: str = "topk", llmint8_params: Optional[list] = None):
        self.num_links = num_links
        self.default_eta = default_eta
        self.method = method
        self.llmint8_params = llmint8_params

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None) -> Dict[str, Any]:
        eta = np.full(self.num_links, self.default_eta)
        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
        }


class UniformCSIOptimizer:
    """
    Uniform compression baseline: channel-aware but not per-link.

    Computes a single eta bottlenecked by the worst link:
        eta = max(eta_min, min(1, min_i(c_i(t) / (R_t * a_i))))

    All links get the same eta. This is the fair comparison against
    CSI-aware per-link allocation — it uses the same measurements
    but cannot differentiate across links.
    """

    def __init__(self, num_links: int, R_t: float, method: str = "topk",
                 eta_min: float = 0.05, llmint8_params: Optional[list] = None):
        self.num_links = num_links
        self.R_t = R_t
        self.method = method
        self.eta_min = eta_min
        self.llmint8_params = llmint8_params

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None) -> Dict[str, Any]:
        uniform_eta = 1.0

        if a_i is not None:
            per_link = []
            for i in range(self.num_links):
                if a_i[i] > 0 and c_t[i] > 0:
                    per_link.append(c_t[i] / (self.R_t * a_i[i]))
                else:
                    per_link.append(1.0)  # no measurement yet

            # Bottlenecked by the worst link
            uniform_eta = min(1.0, min(per_link))
            uniform_eta = max(uniform_eta, self.eta_min)

        eta = np.full(self.num_links, uniform_eta)
        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
        }


# ============================================================================
# CSI-Aware Optimizer — closed-form per-link eta
# ============================================================================

class CSIAwareOptimizer:
    """
    CSI-aware single-task closed-form optimizer.

    Implements:
        eta_i*(t) = min(1, c_i(t) / (R_t * a_i))

    where:
        c_i(t) = measured link throughput (bytes/sec)
        R_t    = target throughput (tokens/sec), so deadline t_0 = 1/R_t
        a_i    = uncompressed activation size at split point i (bytes)

    Example:
        R_t = 2.0 tokens/sec  =>  t_0 = 0.5 sec per token
        If link i has C_i = 150,000 bytes/sec and O_i = 200,000 bytes:
            eta_i = min(1, 150000 / (2.0 * 200000)) = 0.375
        If link j is local with C_j = 800,000 bytes/sec:
            eta_j = min(1, 800000 / (2.0 * 200000)) = 1.0
    """

    def __init__(self, num_links: int, R_t: float, method: str = "topk",
                 eta_min: float = 0.05, llmint8_params: Optional[list] = None):
        """
        Parameters
        ----------
        num_links : int
            Number of inter-node links.
        R_t : float
            Target throughput in tokens/sec. Deadline t_0 = 1/R_t.
        method : str
            Compression method.
        eta_min : float
            Minimum eta to avoid zero-compression edge cases.
        llmint8_params : list, optional
            LLMInt8 parameters if method is llmint8.
        """
        self.num_links = num_links
        self.R_t = R_t
        self.t_0 = 1.0 / R_t
        self.method = method
        self.eta_min = eta_min
        self.llmint8_params = llmint8_params

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute per-link eta from the CSI-aware closed-form formula.

        Parameters
        ----------
        t : int
            Time slot index.
        c_t : np.ndarray [num_links]
            Measured channel capacity per link (bytes/sec).
        a_i : np.ndarray [num_links]
            Uncompressed activation size per link (bytes).
            If None or contains zeros, falls back to eta=1.0 for those links.

        Returns
        -------
        dict with eta, s_comp, codec, codec_params.
        """
        eta = np.ones(self.num_links)

        if a_i is not None:
            for i in range(self.num_links):
                if a_i[i] > 0 and c_t[i] > 0:
                    # eta_i* = min(1, c_i(t) / (R_t * a_i))
                    eta[i] = min(1.0, c_t[i] / (self.R_t * a_i[i]))
                    eta[i] = max(eta[i], self.eta_min)
                # else: keep eta=1.0 (no measurement yet, don't compress)

        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
        }

# ============================================================================
# Channel Estimators — produce c_hat from history
# ============================================================================

class ChannelEstimator:
    """Base class for channel estimation strategies."""

    def __init__(self, num_links: int):
        self.num_links = num_links
        self.history: List[np.ndarray] = []  # list of c_t vectors

    def update(self, c_t: np.ndarray):
        """Record a new true channel observation."""
        self.history.append(c_t.copy())

    def estimate(self) -> np.ndarray:
        """Return c_hat for the current slot."""
        raise NotImplementedError


class MyopicEstimator(ChannelEstimator):
    """
    c_hat(t) = c(t-1): last observation as estimate.
    This is what CSI-aware effectively does.
    """
    def estimate(self) -> np.ndarray:
        if not self.history:
            return np.ones(self.num_links)
        return self.history[-1].copy()


class ConservativeEstimator(ChannelEstimator):
    """
    c_hat(t) = running min of all past observations.
    Pessimistic: always underestimates capacity.
    """
    def estimate(self) -> np.ndarray:
        if not self.history:
            return np.ones(self.num_links)
        stacked = np.stack(self.history)
        # Replace zeros with large number so min ignores uninitialized links
        stacked[stacked == 0] = np.inf
        result = np.min(stacked, axis=0)
        result[result == np.inf] = 1.0  # fallback for links never observed
        return result


class MovingAverageEstimator(ChannelEstimator):
    """
    c_hat(t) = moving average of past W observations.
    Smooths transient fluctuations.
    """
    def __init__(self, num_links: int, window: int = 10):
        super().__init__(num_links)
        self.window = window

    def estimate(self) -> np.ndarray:
        if not self.history:
            return np.ones(self.num_links)
        recent = self.history[-self.window:]
        stacked = np.stack(recent)
        # Average only nonzero entries per link
        result = np.zeros(self.num_links)
        for i in range(self.num_links):
            vals = stacked[:, i]
            nonzero = vals[vals > 0]
            result[i] = nonzero.mean() if len(nonzero) > 0 else 1.0
        return result


class LCBEstimator(ChannelEstimator):
    """
    Lower Confidence Bound (LCB) estimator used by OURS-NOCSI:

        c_hat(t) = mu_hat(t) - beta * sigma_hat(t)

    where mu_hat and sigma_hat are the empirical mean and standard deviation of
    the per-link channel over the last W observations. beta sets the conservatism
    (beta ~= 1.28 gives a one-sided 90% lower confidence bound under a Gaussian).
    By deliberately under-predicting capacity in proportion to its volatility, it
    builds a safety margin against channel drops while staying less pessimistic
    than the running-minimum (Conservative) estimator.
    """
    def __init__(self, num_links: int, beta: float = 1.28, window: int = 10):
        super().__init__(num_links)
        self.beta = beta
        self.window = window

    def estimate(self) -> np.ndarray:
        if not self.history:
            return np.ones(self.num_links)
        recent = self.history[-self.window:]
        stacked = np.stack(recent)
        result = np.zeros(self.num_links)
        for i in range(self.num_links):
            vals = stacked[:, i]
            nonzero = vals[vals > 0]
            if len(nonzero) == 0:
                result[i] = 1.0                      # never observed yet
            elif len(nonzero) == 1:
                result[i] = float(nonzero[0])        # no variance from one sample
            else:
                mu = float(nonzero.mean())
                sigma = float(nonzero.std())         # population std over the window
                # LCB, floored to stay a valid positive capacity (deep pessimism
                # then just maps to eta_min in the optimizer).
                result[i] = max(mu - self.beta * sigma, 1e-9)
        return result


# ============================================================================
# No-CSI Single Task Optimizer — Estimated Stochastic Dual Descent
# ============================================================================

class NoCSISingleTaskOptimizer:
    """
    Algorithm 1: Single-Task Estimated Stochastic Dual Descent.

    Each slot:
      1. Estimate c_hat from channel history
      2. Solve: eta = argmax_{eta} ( A(eta) - mu * lambda * D_hat(eta, c_hat) )
         using epigraph reformulation with scipy SLSQP
      3. Execute pipeline, observe D_actual
      4. Update dual: lambda = max(epsilon, lambda + D_actual - 1/R)

    A(eta) can be:
      - A pre-trained accuracy estimator (from accuracy_estimator.py)
      - A simple proxy: A(eta) = mean(eta)  (default)
    """

    def __init__(self, num_links: int, R_t: float, mu: float = 1.0,
                 epsilon: float = 0.1, method: str = "topk",
                 eta_min: float = 0.05, tau: Optional[np.ndarray] = None,
                 accuracy_fn: Optional[Any] = None, grad_fn: Optional[Any] = None,
                 estimator_type: str = "myopic", ma_window: int = 10,
                 lcb_beta: float = 1.28,
                 llmint8_params: Optional[list] = None):
        """
        Parameters
        ----------
        num_links : int
        R_t : float
            Target throughput (tokens/sec). Deadline t_0 = 1/R_t.
        mu : float
            Tradeoff parameter. Larger = stricter delay compliance, lower accuracy.
        epsilon : float
            Lower bound on dual variable.
        method : str
            Compression method.
        eta_min : float
            Minimum compression ratio per link.
        tau : np.ndarray, optional
            Per-node compute times. If None, estimated from metrics.
        accuracy_fn : callable, optional
            A(eta) -> float. If None, uses A(eta) = mean(eta).
        estimator_type : str
            Channel estimator: "myopic", "conservative", "moving_average", "lcb".
            OURS-NOCSI uses "lcb".
        ma_window : int
            Window size for the moving-average and LCB estimators.
        lcb_beta : float
            LCB conservatism: c_hat = mu - beta*sigma; ~1.28 = one-sided 90%.
        llmint8_params : list, optional
        """
        self.num_links = num_links
        self.R_t = R_t
        self.t_0 = 1.0 / R_t
        self.mu = mu
        self.epsilon = epsilon
        self.lambda_t = epsilon
        self.method = method
        self.eta_min = eta_min
        self.tau_profile = tau  # will be updated from metrics if None
        self.llmint8_params = llmint8_params

        # Accuracy function
        if accuracy_fn is not None:
            self.accuracy_fn = accuracy_fn
            # Use the model's analytic/central-difference gradient if provided,
            # otherwise _accuracy_gradient() falls back to forward finite differences.
            self.grad_fn = grad_fn
        else:
            # Default proxy: A(eta) = mean(eta)
            self.accuracy_fn = lambda eta: float(np.mean(eta))
            self.grad_fn = lambda eta: np.ones(len(eta)) / len(eta)

        # Channel estimator
        if estimator_type == "conservative":
            self.ch_estimator = ConservativeEstimator(num_links)
        elif estimator_type == "moving_average":
            self.ch_estimator = MovingAverageEstimator(num_links, ma_window)
        elif estimator_type == "lcb":
            self.ch_estimator = LCBEstimator(num_links, beta=lcb_beta, window=ma_window)
        else:
            self.ch_estimator = MyopicEstimator(num_links)

        self.estimator_type = estimator_type

    def _accuracy_gradient(self, eta: np.ndarray) -> np.ndarray:
        """Gradient of A(eta), finite differences if no analytical gradient."""
        if self.grad_fn is not None:
            return self.grad_fn(eta)
        # Finite differences
        h = 1e-4
        grad = np.zeros_like(eta)
        f0 = self.accuracy_fn(eta)
        for i in range(len(eta)):
            eta_p = eta.copy()
            eta_p[i] = min(1.0, eta_p[i] + h)
            grad[i] = (self.accuracy_fn(eta_p) - f0) / h
        return grad

    def _objective(self, x: np.ndarray, c_hat: np.ndarray) -> tuple:
        """
        Epigraph objective: min -A(eta) + mu * lambda * z
        where z >= max(tau_i, a_i*eta_i/c_hat_i) for all i.
        """
        eta = x[:-1]
        z = x[-1]

        acc = self.accuracy_fn(eta)
        grad_acc = self._accuracy_gradient(eta)

        obj = -acc + self.mu * self.lambda_t * z

        grad = np.zeros_like(x)
        grad[:-1] = -grad_acc
        grad[-1] = self.mu * self.lambda_t

        return obj, grad

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None,
                 tau: Optional[dict] = None) -> Dict[str, Any]:
        """
        Primal step: solve for eta using estimated channel.

        Parameters
        ----------
        t : int
        c_t : np.ndarray
            True measured channel (used to update estimator history).
        a_i : np.ndarray
            Uncompressed activation sizes per link.
        tau : dict
            Per-node compute times from metrics.
        """
        # Update channel estimator with true observation
        self.ch_estimator.update(c_t)

        # Get estimated channel
        c_hat = self.ch_estimator.estimate()

        # Update tau profile from metrics if available
        if tau is not None and self.tau_profile is None:
            tau_vals = sorted(tau.items())
            self.tau_profile = np.array([v for _, v in tau_vals])

        # Fallback if we don't have a_i or tau yet
        if a_i is None or np.all(a_i == 0) or self.tau_profile is None:
            # Can't solve properly, fall back to CSI-aware with c_hat
            eta = np.ones(self.num_links)
            for i in range(self.num_links):
                if c_hat[i] > 0 and a_i is not None and a_i[i] > 0:
                    eta[i] = min(1.0, c_hat[i] / (self.R_t * a_i[i]))
                    eta[i] = max(eta[i], self.eta_min)
            return {
                "eta": eta,
                "s_comp": np.ones(self.num_links + 1),
                "codec": self.method,
                "codec_params": self.llmint8_params,
                "c_hat": c_hat,
                "lambda": self.lambda_t,
            }

        n = self.num_links
        max_tau = float(np.max(self.tau_profile))

        # Bounds: eta_min <= eta_i <= 1, z >= max_tau
        bounds = [(self.eta_min, 1.0)] * n + [(max_tau, None)]
        x0 = np.array([self.eta_min] * n + [max_tau * 1.1])

        # Constraints: z >= a_i * eta_i / c_hat_i for all links with valid data
        constraints = []
        for i in range(n):
            if c_hat[i] > 0 and a_i[i] > 0:
                beta = a_i[i] / c_hat[i]

                def make_constraint(idx=i, b=beta):
                    return {
                        'type': 'ineq',
                        'fun': lambda x, idx=idx, b=b: x[-1] - b * x[idx],
                        'jac': lambda x, idx=idx, b=b: np.array(
                            [-b if j == idx else (1.0 if j == len(x)-1 else 0.0)
                             for j in range(len(x))])
                    }
                constraints.append(make_constraint())

        try:
            result = minimize(
                fun=self._objective,
                x0=x0,
                args=(c_hat,),
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            eta = np.clip(result.x[:-1], self.eta_min, 1.0)
        except Exception as e:
            logging.warning(f"No-CSI optimization failed: {e}. Using fallback.")
            eta = np.full(n, self.eta_min)

        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
            "c_hat": c_hat,
            "lambda": self.lambda_t,
        }

    def update_dual(self, D_actual: float):
        """
        Dual step: lambda = max(epsilon, lambda + D_actual - 1/R)
        """
        violation = D_actual - self.t_0
        self.lambda_t = max(self.epsilon, self.lambda_t + violation)


# ============================================================================
# No-CSI Baselines — CSI-aware formula with different channel estimates
# ============================================================================

class MyopicCSIOptimizer(CSIAwareOptimizer):
    """
    No-CSI Baseline: myopic.
    Uses CSI-aware formula with c_hat = c(t-1).
    No dual variable — just closed-form with lagged channel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ch_estimator = MyopicEstimator(self.num_links)

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None) -> Dict[str, Any]:
        self.ch_estimator.update(c_t)
        c_hat = self.ch_estimator.estimate()
        # Use parent's CSI-aware formula with c_hat instead of c_t
        eta = np.ones(self.num_links)
        if a_i is not None:
            for i in range(self.num_links):
                if a_i[i] > 0 and c_hat[i] > 0:
                    eta[i] = min(1.0, c_hat[i] / (self.R_t * a_i[i]))
                    eta[i] = max(eta[i], self.eta_min)
        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
            "c_hat": c_hat,
        }


class ConservativeCSIOptimizer(CSIAwareOptimizer):
    """
    No-CSI Baseline: conservative.
    Uses CSI-aware formula with c_hat = running min of past observations.
    Pessimistic — over-compresses to guarantee deadline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ch_estimator = ConservativeEstimator(self.num_links)

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None) -> Dict[str, Any]:
        self.ch_estimator.update(c_t)
        c_hat = self.ch_estimator.estimate()
        eta = np.ones(self.num_links)
        if a_i is not None:
            for i in range(self.num_links):
                if a_i[i] > 0 and c_hat[i] > 0:
                    eta[i] = min(1.0, c_hat[i] / (self.R_t * a_i[i]))
                    eta[i] = max(eta[i], self.eta_min)
        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
            "c_hat": c_hat,
        }


class MovingAvgCSIOptimizer(CSIAwareOptimizer):
    """
    No-CSI Baseline: moving average.
    Uses CSI-aware formula with c_hat = moving average of past W observations.
    """

    def __init__(self, *args, ma_window: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.ch_estimator = MovingAverageEstimator(self.num_links, ma_window)

    def optimize(self, t: int, c_t: np.ndarray, a_i: np.ndarray = None) -> Dict[str, Any]:
        self.ch_estimator.update(c_t)
        c_hat = self.ch_estimator.estimate()
        eta = np.ones(self.num_links)
        if a_i is not None:
            for i in range(self.num_links):
                if a_i[i] > 0 and c_hat[i] > 0:
                    eta[i] = min(1.0, c_hat[i] / (self.R_t * a_i[i]))
                    eta[i] = max(eta[i], self.eta_min)
        return {
            "eta": eta,
            "s_comp": np.ones(self.num_links + 1),
            "codec": self.method,
            "codec_params": self.llmint8_params,
            "c_hat": c_hat,
        }


# ============================================================================
# Directive Broadcaster — sends per-node eta via UDP
# ============================================================================

class DirectiveBroadcaster:
    """
    Sends ControlDirective (per-link eta) to each machine cut-point node via UDP.
    Each cut-point node listens on control_port = base_port + layer_id.
    """

    def __init__(self, node_addrs: Dict[int, tuple], link_of_layer: Dict[int, int]):
        self.node_addrs = node_addrs          # cut-point layer_id -> (ip, port)
        self.link_of_layer = link_of_layer    # cut-point layer_id -> link index
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def broadcast(self, eta: np.ndarray, codec: str, codec_params: Optional[list] = None):
        """
        Send each cut-point node the eta for its link.
        Layer receives: {"eta_i": float, "codec": str, "codec_params": list|null}
        """
        for layer_id, addr in self.node_addrs.items():
            link_idx = self.link_of_layer.get(layer_id)
            eta_i = (float(eta[link_idx])
                     if link_idx is not None and link_idx < len(eta) else 1.0)

            msg = json.dumps({
                "eta_i": eta_i,
                "codec": codec,
                "codec_params": codec_params,
            })
            try:
                self._sock.sendto(msg.encode("utf-8"), addr)
            except OSError as e:
                logging.warning(f"Failed to send directive to layer {layer_id}: {e}")

    def close(self):
        self._sock.close()


# ============================================================================
# Node list parser
# ============================================================================

def parse_node_list(path: str):
    """
    Parse the same node_list.txt used by init_demo.sh.
    Format per line: username-ip-layer1,layer2,...
    Returns dict: layer_id -> ip
    """
    layer_to_ip = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("-")
            ip = parts[1].strip()
            layers = parts[2].strip().split(",")
            for l in layers:
                layer_to_ip[int(l.strip())] = ip
    return layer_to_ip


# ============================================================================
# Main control loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Central controller")
    parser.add_argument("--nodes", type=str, required=True,
                        help="Path to node_list.txt")
    parser.add_argument("--port", type=int, default=9999,
                        help="UDP port for receiving metrics")
    parser.add_argument("--control_base_port", type=int, default=10000,
                        help="Base port for sending directives (node i listens on base+i)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind metric listener")
    parser.add_argument("--algorithm", type=str, default="uniform",
                        help="Optimizer: uniform | uniform_csi | csi_aware | "
                             "no_csi | no_csi_myopic | no_csi_conservative | no_csi_moving_avg")
    parser.add_argument("--R_t", type=float, default=1.0,
                        help="Target throughput in tokens/sec (deadline t_0 = 1/R_t)")
    parser.add_argument("--eta_min", type=float, default=0.05,
                        help="Minimum eta per link")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="No-CSI: tradeoff parameter (larger = stricter delay)")
    parser.add_argument("--dual_epsilon", type=float, default=0.1,
                        help="No-CSI: lower bound on dual variable lambda")
    parser.add_argument("--ma_window", type=int, default=10,
                        help="Window size for the moving-average and LCB estimators")
    parser.add_argument("--lcb_beta", type=float, default=1.28,
                        help="LCB conservatism for no_csi (OURS-NOCSI): "
                             "c_hat = mu - beta*sigma; ~1.28 = one-sided 90%% (default)")
    parser.add_argument("--accuracy_model", type=str, default="outputs/accuracy_model.pkl",
                        help="Path to a fitted accuracy model .pkl (used by no_csi). "
                             "If the file is missing/unloadable, falls back to A(eta)=mean(eta).")
    parser.add_argument("--use_probed", action="store_true",
                        help="Use probed channel capacity instead of retrospective measurement")
    parser.add_argument("--method", type=str, default="topk",
                        help="Compression method: topk | quantization | llmint8 | none")
    parser.add_argument("--default_eta", type=float, default=1.0,
                        help="Default uniform eta (used by uniform optimizer)")
    parser.add_argument("--llmint8_outlier_ratio", type=float, default=0.01,
                        help="LLMInt8: outlier fraction")
    parser.add_argument("--llmint8_outlier_prec", type=str, default="fp16",
                        help="LLMInt8: outlier precision (fp16 | int8)")
    parser.add_argument("--llmint8_regular_prec", type=str, default="int8",
                        help="LLMInt8: regular precision (fp16 | int8 | int4 | int2)")
    parser.add_argument("--slot_interval", type=float, default=2.0,
                        help="Seconds between optimizer invocations")
    parser.add_argument("--output_dir", type=str, default="results/experiment",
                        help="Directory to save metrics (same as batch_inference --output_dir)")
    args = parser.parse_args()

    # Parse node topology
    layer_to_ip = parse_node_list(args.nodes)
    all_layers = sorted(layer_to_ip.keys())
    pipeline_layers = [l for l in all_layers if l > 0]  # exclude UE (layer 0)
    num_pipeline_nodes = len(pipeline_layers)
    first_pipeline_node = pipeline_layers[0]
    last_pipeline_node = pipeline_layers[-1]
    max_layer = all_layers[-1]

    # ---- Machine cut-points ----
    # A pipeline layer is a compression cut-point if its next layer (layer+1,
    # wrapping to the UE at layer 0) is on a DIFFERENT machine, excluding the
    # return hop to the UE/laptop (next layer 0). These are the only links where
    # the activation crosses the network, so they are the only links we optimize
    # (6 inter-Pi links for scripts/nodes.txt: layers 3,5,8,10,12,15).
    cutpoint_layers = []
    for l in pipeline_layers:
        nxt = (l + 1) % (max_layer + 1)   # next node; last pipeline layer wraps to 0 (UE)
        if nxt != 0 and layer_to_ip[l] != layer_to_ip.get(nxt):
            cutpoint_layers.append(l)
    num_links = len(cutpoint_layers)
    link_of_layer = {layer: idx for idx, layer in enumerate(cutpoint_layers)}

    # Machine grouping (pipeline layers per host) for per-machine compute time.
    machine_layers = defaultdict(list)
    for l in pipeline_layers:
        machine_layers[layer_to_ip[l]].append(l)

    logging.info(f"Topology: {num_pipeline_nodes} pipeline layers "
                 f"({first_pipeline_node}-{last_pipeline_node}) on {len(machine_layers)} "
                 f"machines; {num_links} cut-point links")
    logging.info(f"Cut-point layers (compress here): {cutpoint_layers}")
    logging.info(f"Link map (layer -> link idx): {link_of_layer}")
    logging.info(f"Machines (host -> layers): {dict(machine_layers)}")

    # Directive addresses ONLY for cut-point layers (non-cut-points don't listen).
    node_addrs = {}
    for layer_id in cutpoint_layers:
        node_addrs[layer_id] = (layer_to_ip[layer_id], args.control_base_port + layer_id)
    logging.info(f"Directive targets (cut-points): {node_addrs}")

    # Initialize components
    collector = MetricCollector(args.host, args.port)

    llmint8_params = None
    if args.method.lower() in ("llmint8", "llm_int8", "llm.int8"):
        llmint8_params = [
            args.llmint8_outlier_ratio,
            args.llmint8_outlier_prec,
            args.llmint8_regular_prec,
        ]

    if args.algorithm == "csi_aware":
        optimizer = CSIAwareOptimizer(
            num_links, R_t=args.R_t, method=args.method,
            eta_min=args.eta_min, llmint8_params=llmint8_params)
        logging.info(f"CSI-aware optimizer: R_t={args.R_t}, t_0={1.0/args.R_t:.3f}s")
    elif args.algorithm == "uniform_csi":
        optimizer = UniformCSIOptimizer(
            num_links, R_t=args.R_t, method=args.method,
            eta_min=args.eta_min, llmint8_params=llmint8_params)
        logging.info(f"Uniform-CSI optimizer: R_t={args.R_t}, t_0={1.0/args.R_t:.3f}s")
    elif args.algorithm == "no_csi":
        # Load the fitted accuracy surrogate A(eta) if available; otherwise the
        # optimizer falls back to A(eta) = mean(eta).
        accuracy_fn, grad_fn = None, None
        if args.accuracy_model and os.path.exists(args.accuracy_model):
            try:
                from accuracy_model import load_accuracy_model
                acc_model = load_accuracy_model(args.accuracy_model)
                if getattr(acc_model, "is_fitted", False):
                    accuracy_fn = acc_model.predict
                    grad_fn = acc_model.gradient
                    logging.info(f"Loaded accuracy model '{args.accuracy_model}' "
                                 f"(type={acc_model.model_type}) -> using A(eta)=model")
                else:
                    logging.warning(f"Accuracy model '{args.accuracy_model}' is not fitted; "
                                    f"using A(eta)=mean(eta)")
            except Exception as e:
                logging.warning(f"Could not load accuracy model '{args.accuracy_model}': {e}; "
                                f"using A(eta)=mean(eta)")
        else:
            logging.info(f"No accuracy model at '{args.accuracy_model}'; using A(eta)=mean(eta)")

        optimizer = NoCSISingleTaskOptimizer(
            num_links, R_t=args.R_t, mu=args.mu, epsilon=args.dual_epsilon,
            method=args.method, eta_min=args.eta_min,
            accuracy_fn=accuracy_fn, grad_fn=grad_fn,
            estimator_type="lcb", ma_window=args.ma_window, lcb_beta=args.lcb_beta,
            llmint8_params=llmint8_params)
        logging.info(f"No-CSI optimizer (OURS-NOCSI): R_t={args.R_t}, mu={args.mu}, "
                     f"epsilon={args.dual_epsilon}, estimator=lcb(beta={args.lcb_beta}, "
                     f"window={args.ma_window}), A(eta)={'model' if accuracy_fn else 'mean(eta)'}")
    elif args.algorithm == "no_csi_myopic":
        optimizer = MyopicCSIOptimizer(
            num_links, R_t=args.R_t, method=args.method,
            eta_min=args.eta_min, llmint8_params=llmint8_params)
        logging.info(f"No-CSI Myopic baseline: R_t={args.R_t}")
    elif args.algorithm == "no_csi_conservative":
        optimizer = ConservativeCSIOptimizer(
            num_links, R_t=args.R_t, method=args.method,
            eta_min=args.eta_min, llmint8_params=llmint8_params)
        logging.info(f"No-CSI Conservative baseline: R_t={args.R_t}")
    elif args.algorithm == "no_csi_moving_avg":
        optimizer = MovingAvgCSIOptimizer(
            num_links, R_t=args.R_t, method=args.method,
            eta_min=args.eta_min, ma_window=args.ma_window,
            llmint8_params=llmint8_params)
        logging.info(f"No-CSI Moving Average baseline: R_t={args.R_t}, window={args.ma_window}")
    else:
        optimizer = UniformOptimizer(
            num_links, args.default_eta, args.method,
            llmint8_params=llmint8_params)
        logging.info(f"Uniform optimizer: eta={args.default_eta}")

    broadcaster = DirectiveBroadcaster(node_addrs, link_of_layer)

    collector.start()
    logging.info("Controller running. Press Ctrl+C to stop.")

    step = 0
    per_slot_records: List[Dict[str, Any]] = []  # per-slot {step, D_act, excess_delay, eta}
    cumulative_excess_delay = 0.0
    cumulative_d_act = 0.0
    valid_delay_slots = 0
    try:
        while True:
            time.sleep(args.slot_interval)

            state = collector.get_system_state(
                num_links, link_of_layer,
                first_pipeline_node=first_pipeline_node,
                last_pipeline_node=last_pipeline_node)
            if state is None:
                logging.info(f"[slot {step}] Waiting for node reports...")
                step += 1
                continue

            c_t = state["c_t"]
            c_probed = state["c_probed"]
            tau = state["tau"]
            a_i = state["a_i"]
            d_actual_wall = state["D_actual"]

            # Select channel measurement: probed or retrospective
            if args.use_probed:
                c_used = np.where(c_probed > 0, c_probed, c_t)
            else:
                c_used = c_t

            # Log current state
            print()
            logging.info(f"[slot {step}] C(t)  = {c_t}")
            logging.info(f"[slot {step}] a_i   = {a_i}")
            logging.info(f"[slot {step}] tau   = {tau}")
            if d_actual_wall is not None:
                logging.info(f"[slot {step}] D_actual (wall) = {d_actual_wall:.4f}s")

            # Run optimizer
            if isinstance(optimizer, NoCSISingleTaskOptimizer):
                result = optimizer.optimize(t=step, c_t=c_used, a_i=a_i, tau=tau)
            else:
                result = optimizer.optimize(t=step, c_t=c_used, a_i=a_i)
            eta = result["eta"]
            codec = result["codec"]

            logging.info(f"[slot {step}] eta*  = {eta}")
            logging.info(f"[slot {step}] codec = {codec}")

            # Log No-CSI specific info
            if "c_hat" in result:
                logging.info(f"[slot {step}] c_hat = {result['c_hat']}")
            if "lambda" in result:
                logging.info(f"[slot {step}] lambda = {result['lambda']:.4f}")

            # ----------------------------------------------------------
            # D_act(t) = max( max_machine τ_machine, max_link (a_i * η_i / c_i) )
            # τ_machine = sum of the compute times of the layers on that machine
            # (the pipeline stage is a machine); comm is over the cut-point links.
            # ----------------------------------------------------------
            machine_tau = defaultdict(float)
            for node_id, t_i in tau.items():
                ip = layer_to_ip.get(node_id)
                if ip is not None:
                    machine_tau[ip] += t_i
            max_tau = max(machine_tau.values()) if machine_tau else 0.0

            max_comm = 0.0
            for i in range(num_links):
                if c_t[i] > 0 and a_i[i] > 0:
                    comm_delay = (a_i[i] * eta[i]) / c_t[i]
                    max_comm = max(max_comm, comm_delay)

            d_actual_formula = max(max_tau, max_comm)

            logging.info(f"[slot {step}] D_act (formula) = {d_actual_formula:.4f}s  "
                         f"[max_tau={max_tau:.4f}, max_comm={max_comm:.4f}]")

            R_t = getattr(optimizer, 'R_t', args.R_t)
            t_0 = 1.0 / R_t

            # ----------------------------------------------------------
            # Excess delay: D_act(t) - 1/R(t)
            # Every slot counts — we record and use all slots (prefill + decode).
            # ----------------------------------------------------------
            excess_delay_t = d_actual_formula - t_0

            cumulative_excess_delay += excess_delay_t
            cumulative_d_act += d_actual_formula
            valid_delay_slots += 1
            avg_excess_delay = cumulative_excess_delay / valid_delay_slots
            avg_d_act = cumulative_d_act / valid_delay_slots

            logging.info(f"[slot {step}] excess_delay(t) = {excess_delay_t:.4f}s  "
                         f"(<=0 means deadline met, t_0={t_0:.4f}s)")
            logging.info(f"[slot {step}] avg_excess_delay = {avg_excess_delay:.4f}s  "
                         f"avg_D_act = {avg_d_act:.4f}s  "
                         f"({valid_delay_slots} slots)")

            # --- No-CSI dual update: lambda = max(eps, lambda + D_act - t_0) ---
            if hasattr(optimizer, 'update_dual'):
                optimizer.update_dual(d_actual_formula)
                logging.info(f"[slot {step}] lambda_updated = {optimizer.lambda_t:.4f}")

            # Record per-slot delay breakdown (dumped to disk on shutdown)
            per_slot_records.append({
                "step": step,
                "D_act": round(float(d_actual_formula), 6),
                "excess_delay": round(float(excess_delay_t), 6),
                "eta": [round(float(e), 6) for e in eta],
                "t_0": round(float(t_0), 6),
            })

            # Broadcast to nodes
            broadcaster.broadcast(eta, codec, result.get("codec_params"))

            step += 1

    except KeyboardInterrupt:
        logging.info("Shutting down...")
        os.makedirs(args.output_dir, exist_ok=True)

        if valid_delay_slots > 0:
            avg_d_act = cumulative_d_act / valid_delay_slots
            avg_excess_delay = cumulative_excess_delay / valid_delay_slots
            logging.info(f"=== FINAL SUMMARY ===")
            logging.info(f"Total slots: {valid_delay_slots}")
            logging.info(f"Avg D_act (formula): {avg_d_act:.4f}s")
            logging.info(f"Avg excess delay: {avg_excess_delay:.4f}s  "
                         f"({'MEETING DEADLINE' if avg_excess_delay <= 0 else 'VIOLATING DEADLINE'})")
            logging.info(f"Target: R_t={R_t} tok/s, t_0={1.0/R_t:.4f}s")
            if hasattr(optimizer, 'lambda_t'):
                logging.info(f"Final lambda: {optimizer.lambda_t:.4f}  "
                             f"(mu={args.mu}, epsilon={args.dual_epsilon})")

            # Save summary to output_dir
            summary = {
                "algorithm": args.algorithm,
                "method": args.method,
                "R_t": float(R_t),
                "t_0": float(1.0 / R_t),
                "eta_min": float(args.eta_min),
                "total_slots": int(valid_delay_slots),
                "avg_D_act": round(float(avg_d_act), 6),
                "avg_excess_delay": round(float(avg_excess_delay), 6),
                "deadline_met": bool(avg_excess_delay <= 0),
            }
            # Add No-CSI specific fields
            if hasattr(optimizer, 'lambda_t'):
                summary["final_lambda"] = round(float(optimizer.lambda_t), 6)
                summary["mu"] = float(args.mu)
                summary["dual_epsilon"] = float(args.dual_epsilon)
            if hasattr(optimizer, 'estimator_type'):
                summary["estimator_type"] = optimizer.estimator_type
            summary_path = os.path.join(args.output_dir, "controller_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logging.info(f"Summary saved to {summary_path}")
        else:
            logging.info("No slots collected.")

    finally:
        collector.stop()
        broadcaster.close()

        # Dump full metric history to output_dir
        if collector.history:
            os.makedirs(args.output_dir, exist_ok=True)
            history_path = os.path.join(args.output_dir, "metrics_log.json")
            with open(history_path, "w") as f:
                json.dump(collector.history, f, indent=2)
            logging.info(f"Saved {len(collector.history)} metric records to {history_path}")

        # Dump per-slot delay breakdown (step, D_act, excess_delay, eta)
        if per_slot_records:
            os.makedirs(args.output_dir, exist_ok=True)
            slot_path = os.path.join(args.output_dir, "per_slot_delays.json")
            with open(slot_path, "w") as f:
                json.dump(per_slot_records, f, indent=2)
            logging.info(f"Saved {len(per_slot_records)} per-slot delay records to {slot_path}")

            # Also write a flat CSV for quick spreadsheet/plotting use
            slot_csv = os.path.join(args.output_dir, "per_slot_delays.csv")
            max_links = max(len(r["eta"]) for r in per_slot_records)
            with open(slot_csv, "w") as f:
                eta_cols = ",".join(f"eta_{i}" for i in range(max_links))
                f.write(f"step,D_act,excess_delay,t_0,{eta_cols}\n")
                for r in per_slot_records:
                    eta_vals = r["eta"] + [""] * (max_links - len(r["eta"]))
                    eta_str = ",".join(str(e) for e in eta_vals)
                    f.write(f"{r['step']},{r['D_act']},{r['excess_delay']},"
                            f"{r['t_0']},{eta_str}\n")
            logging.info(f"Saved per-slot delay CSV to {slot_csv}")


if __name__ == "__main__":
    main()