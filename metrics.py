"""
metrics.py
----------
Link budget calculations and theoretical BER curves.

Provides:
  * Eb/N0 from link parameters
  * Theoretical BER for 16-QAM over AWGN
  * EVM (Error Vector Magnitude) measurement
  * PAPR measurement
"""

import numpy as np
from scipy.special import erfc
from config import Config, K_BOLTZMANN


# ---------------------------------------------------------------------------
# Theoretical BER for square 16-QAM (Gray-coded, AWGN)
# ---------------------------------------------------------------------------

def ber_theory_16qam_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """
    Approximate BER for Gray-coded 16-QAM in AWGN.

    BER ≈ (3/8) * erfc( sqrt(Eb/N0 / 5) )

    This is the standard closed-form approximation (exact for Gray coding).

    Parameters
    ----------
    ebn0_db : 1-D array of Eb/N0 values in dB

    Returns
    -------
    ber : 1-D array of BER values
    """
    ebn0_lin = 10 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    ber = (3.0 / 8.0) * erfc(np.sqrt(ebn0_lin / 5.0))
    return ber


# ---------------------------------------------------------------------------
# Link budget SNR / Eb/N0 calculator
# ---------------------------------------------------------------------------

def compute_ebn0_db(cfg: Config,
                    noise_temp_k: float = None) -> float:
    """
    Compute the theoretical Eb/N0 at the receiver.

    Eb/N0 (dB) = Tx_power_dBW + Gt_dBi - FSPL_dB + Gr_dBi + LNA_dB
                 - 10*log10(k_B * T_sys)
                 - 10*log10(symbol_rate * bits_per_symbol)

    We work in normalised units where Tx power = 0 dBW,
    symbol rate = 1 sym/s, so the result is relative.
    """
    if noise_temp_k is None:
        noise_temp_k = cfg.noise_temp_k

    if noise_temp_k <= 0:
        return float('inf')

    # Net received signal power (dB, relative)
    net_gain_db = (cfg.tx_antenna_gain_dbi
                   - cfg.free_space_path_loss_db
                   + cfg.rx_antenna_gain_dbi
                   + cfg.lna_gain_db)

    # Noise power spectral density N0 = k_B * T_sys
    n0_db = 10 * np.log10(K_BOLTZMANN * noise_temp_k)

    # Bit rate = symbol_rate * bits_per_symbol (= bits_per_symbol for unit sym rate)
    rb_db = 10 * np.log10(float(cfg.bits_per_symbol))

    ebn0_db = net_gain_db - n0_db - rb_db
    return ebn0_db


# ---------------------------------------------------------------------------
# EVM (Error Vector Magnitude)
# ---------------------------------------------------------------------------

def compute_evm(ref_symbols: np.ndarray,
                rx_symbols: np.ndarray) -> float:
    """
    Global burst-level EVM in percent.

    This is the direct Python EVM calculation:
        EVM = RMS(rx - ref) / RMS(ref)

    It is useful as an end-to-end numeric metric, but it does not always match
    MATLAB Constellation Scope values when there is an uncorrected time-varying
    phase rotation, such as Doppler without carrier synchronization.
    """
    n = min(len(ref_symbols), len(rx_symbols))
    if n == 0:
        return float('nan')

    ref = ref_symbols[:n]
    rx = rx_symbols[:n]

    denom = np.mean(np.abs(ref) ** 2)
    if denom <= 0:
        return float('nan')

    err = rx - ref
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / denom)
    return float(evm * 100.0)


def compute_evm_matlab_scope_like(ref_symbols: np.ndarray,
                                  rx_symbols: np.ndarray,
                                  window_size: int = 50,
                                  align_each_window: bool = True) -> float:
    """
    MATLAB Constellation-Scope-like EVM in percent.

    MATLAB's constellation/EVM scopes often report an EVM over a measurement
    interval rather than one global full-burst value. In the uncorrected
    Doppler case, the constellation rotates over time. A full-burst EVM is then
    very large, while a short-window EVM is much closer to the MATLAB Scope
    display.

    This function computes EVM over short windows. By default, each window is
    independently fitted with one complex scalar, which removes only a local
    constant gain/phase offset. This does NOT correct Doppler in the receiver;
    it only makes the reported EVM comparable to MATLAB's scope-style display.

    Parameters
    ----------
    ref_symbols : ideal transmitted symbols
    rx_symbols  : received symbols
    window_size : number of symbols per EVM measurement window
                  50 symbols matches the MATLAB Scope value observed for the
                  Doppler=3 Hz, correction-OFF scenario.
    align_each_window : if True, remove one best complex gain/phase factor
                        independently in each window.

    Returns
    -------
    evm_percent : mean windowed EVM in percent
    """
    n = min(len(ref_symbols), len(rx_symbols))
    if n == 0:
        return float('nan')

    ref = ref_symbols[:n]
    rx = rx_symbols[:n]

    window_size = int(window_size)
    if window_size <= 0 or window_size > n:
        window_size = n

    evm_values = []

    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        ref_w = ref[start:end]
        rx_w = rx[start:end]

        if align_each_window:
            denom = np.vdot(rx_w, rx_w)
            if np.abs(denom) > 1e-30:
                # Best scalar c minimizing ||c*rx_w - ref_w||^2
                c = np.vdot(rx_w, ref_w) / denom
                rx_w = c * rx_w

        evm_values.append(compute_evm(ref_w, rx_w))

    if not evm_values:
        return compute_evm(ref, rx)

    return float(np.mean(evm_values))


# ---------------------------------------------------------------------------
# PAPR
# ---------------------------------------------------------------------------

def compute_papr_db(signal: np.ndarray) -> float:
    """Peak-to-Average Power Ratio in dB."""
    peak = np.max(np.abs(signal) ** 2)
    avg = np.mean(np.abs(signal) ** 2)
    if avg <= 0:
        return 0.0
    return float(10 * np.log10(peak / avg))


# ---------------------------------------------------------------------------
# BER result container
# ---------------------------------------------------------------------------

class ScenarioResult:
    """Store results for one simulation scenario."""

    def __init__(self, name: str):
        self.name = name
        self.ber: float = 1.0
        self.ser: float = 1.0
        self.n_errors: int = 0
        self.ebn0_db: float = 0.0
        self.snr_db: float = 0.0
        # evm_pct is the reported EVM used in the main tables.
        # evm_global_pct keeps the direct full-burst value for diagnostics.
        self.evm_pct: float = 0.0
        self.evm_global_pct: float = 0.0
        self.evm_metric: str = "global"
        self.papr_db: float = 0.0
        self.notes: str = ""

    def __repr__(self):
        return (f"ScenarioResult(name={self.name!r}, "
                f"BER={self.ber:.2e}, EbN0={self.ebn0_db:.1f} dB)")
