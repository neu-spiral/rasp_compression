"""
tools/metrics.py  –  Per-node metrics collection and reporting
===============================================================

Each pipeline node uses NodeMetrics to record per-forward-pass timing,
then periodically reports to the central controller via UDP.

Metrics collected per inference step:
    - tau_i      : forward pass wall-clock time (seconds)
    - tx_time_i  : time to transmit compressed activation to next node (seconds)
    - tx_bytes_i : compressed payload size (bytes)
    - c_i        : effective link throughput = tx_bytes_i / tx_time_i (bytes/sec)
    - t_arrival  : wall-clock time when input arrived at this node
    - t_departure: wall-clock time when output left this node

The central controller reconstructs:
    - D_actual = t_departure(last_node) - t_arrival(first_node)
    - C_i(t) from each node's c_i
"""

import time
import json
import socket
import logging
import threading
from typing import Optional, Dict, Any


PROBE_PAYLOAD_SIZE = 4096  # bytes — roughly matches decode activation size
PROBE_MARKER = b"PROBE"


class LinkProber:
    """
    Background thread that periodically probes the outgoing link
    by sending a small TCP packet and timing the round trip.

    Gives a fresh c_i estimate BEFORE the actual data transfer,
    unlike the retrospective measurement (bytes/time after transfer).

    The probe target must be listening for probe connections.
    We reuse the same port as the next node — the node's receive_data
    handles probe packets separately.
    """

    def __init__(self, target_host: str, target_port: int,
                 probe_interval: float = 2.0, payload_size: int = PROBE_PAYLOAD_SIZE):
        self.target = (target_host, int(target_port))
        self.probe_interval = probe_interval
        self.payload_size = payload_size
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.last_throughput: float = 0.0  # bytes/sec
        self.last_rtt: float = 0.0  # seconds

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._probe_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _probe_loop(self):
        while self._running:
            try:
                self._do_probe()
            except Exception as e:
                logging.debug(f"Probe failed: {e}")
            time.sleep(self.probe_interval)

    def _do_probe(self):
        """Send a small payload, wait for ACK, measure throughput."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            sock.connect(self.target)

            # Build probe payload: PROBE marker + padding
            payload = PROBE_MARKER + b'\x00' * (self.payload_size - len(PROBE_MARKER))

            t_start = time.perf_counter()
            sock.sendall(payload)
            sock.sendall(b"END")

            # Wait for ACK
            ack = sock.recv(32)
            t_end = time.perf_counter()

            rtt = t_end - t_start
            throughput = self.payload_size / rtt if rtt > 1e-9 else 0.0

            with self._lock:
                self.last_throughput = throughput
                self.last_rtt = rtt

        finally:
            sock.close()

    def get_estimate(self) -> float:
        """Return the most recent probed throughput (bytes/sec)."""
        with self._lock:
            return self.last_throughput


class NodeMetrics:
    """
    Lightweight metrics recorder for a single pipeline node.

    Usage in resilientNode.py handle_client():

        metrics.mark_arrival()
        # ... receive data ...
        metrics.mark_compute_start()
        # ... forward pass ...
        metrics.mark_compute_end()
        metrics.mark_tx_start()
        # ... send_data / transmit_data_to_next_nodes ...
        metrics.mark_tx_end(tx_bytes)
        metrics.report()
    """

    def __init__(self, node_id: int, controller_host: str, controller_port: int):
        """
        Parameters
        ----------
        node_id : int
            This node's layer index in the pipeline.
        controller_host : str
            IP address of the central controller.
        controller_port : int
            UDP port the controller listens on for metric reports.
        """
        self.node_id = node_id
        self.controller_addr = (controller_host, controller_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Timestamps (perf_counter for durations, time.time for cross-node sync)
        self._t_arrival: float = 0.0
        self._t_arrival_wall: float = 0.0
        self._t_compute_start: float = 0.0
        self._t_compute_end: float = 0.0
        self._t_tx_start: float = 0.0
        self._t_tx_end: float = 0.0
        self._t_departure_wall: float = 0.0
        self._tx_bytes: int = 0
        self._raw_bytes: int = 0

        self._step: int = 0

        # Link prober (set externally for non-last nodes)
        self.prober: Optional[LinkProber] = None

    def mark_arrival(self):
        """Call when data arrives at this node (before receive_data returns)."""
        self._t_arrival = time.perf_counter()
        self._t_arrival_wall = time.time()

    def mark_compute_start(self):
        """Call right before the forward pass."""
        self._t_compute_start = time.perf_counter()

    def mark_compute_end(self):
        """Call right after the forward pass."""
        self._t_compute_end = time.perf_counter()

    def mark_tx_start(self):
        """Call right before transmitting to next node."""
        self._t_tx_start = time.perf_counter()

    def mark_tx_end(self, tx_bytes: int, raw_bytes: int = 0):
        """Call right after transmit completes.
        tx_bytes  = compressed payload size.
        raw_bytes = uncompressed activation size (O_i).
        """
        self._t_tx_end = time.perf_counter()
        self._t_departure_wall = time.time()
        self._tx_bytes = tx_bytes
        self._raw_bytes = raw_bytes

    def report(self):
        """Send metrics to the central controller via UDP (non-blocking)."""
        tau_i = self._t_compute_end - self._t_compute_start
        tx_time = self._t_tx_end - self._t_tx_start
        c_i = self._tx_bytes / tx_time if tx_time > 1e-9 else 0.0

        # Probed capacity (if prober is attached)
        c_probed = self.prober.get_estimate() if self.prober else 0.0

        msg = {
            "node_id": self.node_id,
            "step": self._step,
            "tau_i": round(tau_i, 6),
            "tx_time_i": round(tx_time, 6),
            "tx_bytes_i": self._tx_bytes,
            "raw_bytes_i": self._raw_bytes,
            "c_i": round(c_i, 2),
            "c_probed_i": round(c_probed, 2),
            "t_arrival_wall": self._t_arrival_wall,
            "t_departure_wall": self._t_departure_wall,
        }

        try:
            self._sock.sendto(json.dumps(msg).encode("utf-8"), self.controller_addr)
        except OSError as e:
            logging.warning(f"Failed to send metrics: {e}")

        self._step += 1

    def close(self):
        self._sock.close()


class LastNodeMetrics(NodeMetrics):
    """
    Variant for the last pipeline node — no transmission, just compute.
    Reports D_actual proxy (t_compute_end_wall - t_arrival_wall).
    """

    def mark_tx_end(self, tx_bytes: int = 0, raw_bytes: int = 0):
        """Last node: no transmission, departure = compute end."""
        self._t_tx_end = self._t_compute_end
        self._t_departure_wall = time.time()
        self._tx_bytes = 0
        self._raw_bytes = 0

    def report(self):
        tau_i = self._t_compute_end - self._t_compute_start

        msg = {
            "node_id": self.node_id,
            "step": self._step,
            "tau_i": round(tau_i, 6),
            "tx_time_i": 0.0,
            "tx_bytes_i": 0,
            "raw_bytes_i": 0,
            "c_i": 0.0,
            "t_arrival_wall": self._t_arrival_wall,
            "t_departure_wall": self._t_departure_wall,
            "is_last_node": True,
        }

        try:
            self._sock.sendto(json.dumps(msg).encode("utf-8"), self.controller_addr)
        except OSError as e:
            logging.warning(f"Failed to send metrics: {e}")

        self._step += 1