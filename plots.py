"""
plots.py
--------
All visualisation functions for the RF Satellite Link simulation.

Plots produced:
  1. Constellation: before HPA / after HPA / received
  2. HPA AM/AM and AM/PM characteristics
  3. Spectrum: transmitted and received signals
  4. BER comparison across scenarios
  5. SRRC filter frequency response
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator
from typing import List, Optional, Dict

from modulation import SYMBOLS_16QAM


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "axes.titlecolor": "#e6edf3",
    "figure.titlesize": 14,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
}

ACCENT = "#58a6ff"
GOLD   = "#f7c948"
TEAL   = "#39d353"
RED    = "#f85149"


def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# 1. Constellation diagram
# ---------------------------------------------------------------------------

def plot_constellations(symbols_before_hpa: np.ndarray,
                        symbols_after_hpa: np.ndarray,
                        symbols_rx: np.ndarray,
                        title_suffix: str = "",
                        max_points: int = 2000,
                        save_path: Optional[str] = None):
    """
    Three-panel constellation: pre-HPA, post-HPA, received.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Constellation Diagrams  {title_suffix}", fontsize=13, color="#e6edf3")

    datasets = [
        (symbols_before_hpa, "Before HPA", ACCENT),
        (symbols_after_hpa,  "After HPA",  GOLD),
        (symbols_rx,         "Received",   TEAL),
    ]

    for ax, (syms, label, color) in zip(axes, datasets):
        # Sub-sample for speed
        n = min(max_points, len(syms))
        idx = np.random.choice(len(syms), n, replace=False)
        s = syms[idx]

        ax.scatter(s.real, s.imag, s=2, alpha=0.4, color=color, rasterized=True)

        # Ideal reference points
        ax.scatter(SYMBOLS_16QAM.real, SYMBOLS_16QAM.imag,
                   s=60, c="white", marker="+", linewidths=1.2, zorder=5)

        lim = max(np.percentile(np.abs(s.real), 99.5),
                  np.percentile(np.abs(s.imag), 99.5)) * 1.3
        if not np.isfinite(lim) or lim <= 0:
            lim = 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#30363d", lw=0.8)
        ax.axvline(0, color="#30363d", lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("In-Phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True)
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. HPA AM/AM and AM/PM curves
# ---------------------------------------------------------------------------

def plot_hpa_characteristics(a_a, b_a, a_p, b_p,
                              ibo_levels: List[float] = (30, 7, 1),
                              save_path: Optional[str] = None):
    """Plot Saleh TWTA AM/AM and AM/PM curves with operating point markers."""
    from impairments import hpa_am_am_curve

    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("HPA Saleh TWTA Characteristics", fontsize=13, color="#e6edf3")

    r_in, r_out = hpa_am_am_curve(a_a, b_a, n_points=400)
    r_sat = 1.0 / np.sqrt(b_a)

    # AM/AM
    ax1.plot(r_in, r_in * a_a, "--", color="#8b949e", lw=1, label="Linear (ideal)")
    ax1.plot(r_in, r_out, color=ACCENT, lw=2, label="Saleh AM/AM")
    ax1.axvline(r_sat, color=RED, lw=1, ls=":", label=f"r_sat={r_sat:.2f}")

    colors = [TEAL, GOLD, RED]
    for ibo_db, col in zip(ibo_levels, colors):
        ibo_lin = 10 ** (ibo_db / 10.0)
        r_op = r_sat / np.sqrt(ibo_lin)
        a_op = a_a * r_op / (1 + b_a * r_op ** 2)
        ax1.scatter([r_op], [a_op], color=col, s=60, zorder=5, label=f"IBO {ibo_db} dB")

    ax1.set_title("AM/AM Characteristic")
    ax1.set_xlabel("Input Amplitude")
    ax1.set_ylabel("Output Amplitude")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # AM/PM
    phi = a_p * r_in ** 2 / (1 + b_p * r_in ** 2)
    ax2.plot(r_in, np.rad2deg(phi), color=GOLD, lw=2)
    ax2.axvline(r_sat, color=RED, lw=1, ls=":", label=f"r_sat={r_sat:.2f}")
    ax2.set_title("AM/PM Characteristic")
    ax2.set_xlabel("Input Amplitude")
    ax2.set_ylabel("Phase Shift (deg)")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Power Spectral Density
# ---------------------------------------------------------------------------

def plot_spectra(tx_signal: np.ndarray,
                 rx_signal: np.ndarray,
                 sample_rate: float,
                 title_suffix: str = "",
                 save_path: Optional[str] = None):
    """
    MATLAB-Scope-like PSD overlay for Tx and Rx.

    The simulator's channel signal can include very large physical link-budget
    attenuation, while MATLAB's Spectrum Analyzer displays Tx/Rx traces on a
    convenient visual scale.  Therefore this plot performs display-only peak
    normalization: it aligns the *shape* of the Tx and Rx spectra on one axes,
    without changing BER/EVM/MER/SNR or any saved numeric results.

    The x-axis is kept in the simulator's native normalized-frequency units
    (symbol-rate units).  Do not divide by 1e3 here unless sample_rate is a real
    Hz sample rate; in this project it is usually an oversampling rate such as 8.
    """
    from scipy.signal import welch

    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"Tx and Rx Spectrum  {title_suffix}", fontsize=13, color="#e6edf3")

    tx_signal = np.asarray(tx_signal)
    rx_signal = np.asarray(rx_signal)

    min_len = min(len(tx_signal), len(rx_signal))
    nperseg = min(2048, max(256, min_len // 8))
    if nperseg > min_len:
        nperseg = min_len

    # Display offsets chosen to resemble MATLAB's overlaid spectrum view:
    # Tx slightly above Rx, both visible on the same axes.
    spectra = [
        (tx_signal, "Transmitted", GOLD,   8.0),
        (rx_signal, "Received",    ACCENT, -15.0),
    ]

    all_y = []
    for sig, label, color, display_peak_db in spectra:
        f, Pxx = welch(
            sig,
            fs=sample_rate,
            nperseg=nperseg,
            return_onesided=False,
            scaling="density",
        )

        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
        Pxx_dB = 10.0 * np.log10(Pxx + 1e-30)

        # Display-only peak normalization.  This fixes the previous issue where
        # path loss made Rx appear ~100 dB below Tx, and also avoids collapsing
        # the frequency axis into a vertical line.
        finite = np.isfinite(Pxx_dB)
        if np.any(finite):
            Pxx_dB = Pxx_dB - np.max(Pxx_dB[finite]) + display_peak_db

        ax.plot(f, Pxx_dB, color=color, lw=1.2, label=label)
        all_y.append(Pxx_dB)

    ax.set_xlabel("Normalised Frequency (sym/s)")
    ax.set_ylabel("Power (dB, display-normalized)")

    # In this project sample_rate is usually samples/symbol, e.g. 8, so the
    # Nyquist range is ±sample_rate/2 = ±4 sym/s.
    f_lim = sample_rate / 2.0
    if np.isfinite(f_lim) and f_lim > 0:
        ax.set_xlim(-f_lim, f_lim)

    if all_y:
        y = np.concatenate([yy[np.isfinite(yy)] for yy in all_y if len(yy) > 0])
        if len(y) > 0:
            y_top = max(20.0, np.percentile(y, 99.5) + 5.0)
            y_bottom = min(-90.0, np.percentile(y, 1.0) - 5.0)
            ax.set_ylim(y_bottom, y_top)

    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)

    fig.text(
        0.5, 0.01,
        "Display note: Tx/Rx spectra are peak-normalized for MATLAB-Scope-like "
        "visual comparison. This affects only this figure, not the simulation metrics.",
        ha="center", va="bottom", fontsize=7.5, color="#8b949e",
        style="italic", wrap=True,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. BER comparison bar / curve
# ---------------------------------------------------------------------------

def plot_ber_comparison(scenario_names: List[str],
                        ber_values: List[float],
                        ebn0_values: Optional[List[float]] = None,
                        title: str = "BER Comparison Across Scenarios",
                        save_path: Optional[str] = None):
    """Horizontal bar chart of BER per scenario on a log scale."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(scenario_names))))
    fig.suptitle(title, fontsize=13, color="#e6edf3")

    y = np.arange(len(scenario_names))
    bers = np.array(ber_values, dtype=float)
    bers = np.clip(bers, 1e-7, 1.0)

    colors = [TEAL if b < 1e-3 else GOLD if b < 1e-1 else RED for b in bers]

    bars = ax.barh(y, bers, color=colors, height=0.55, log=True)
    ax.set_yticks(y)
    ax.set_yticklabels(scenario_names, fontsize=9)
    ax.set_xlabel("Bit Error Rate (BER)")
    ax.set_xlim(1e-7, 1.2)
    ax.axvline(1e-3, color="#8b949e", ls="--", lw=0.8, label="10⁻³ threshold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x")

    # Annotate bars
    for bar, ber in zip(bars, bers):
        ax.text(ber * 1.5, bar.get_y() + bar.get_height() / 2,
                f"{ber:.1e}", va="center", ha="left", fontsize=8, color="#c9d1d9")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Theoretical vs simulated BER curve
# ---------------------------------------------------------------------------

def plot_ber_vs_ebn0(sim_ebn0: List[float],
                     sim_ber: List[float],
                     title: str = "BER vs Eb/N0 – 16-QAM AWGN",
                     save_path: Optional[str] = None):
    """
    Plot simulated BER points against the theoretical 16-QAM AWGN curve.
    """
    from metrics import ber_theory_16qam_awgn

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title, fontsize=13, color="#e6edf3")

    ebn0_range = np.linspace(0, 25, 300)
    ax.semilogy(ebn0_range, ber_theory_16qam_awgn(ebn0_range),
                color=ACCENT, lw=2, label="Theory (16-QAM AWGN)")

    valid = [(e, b) for e, b in zip(sim_ebn0, sim_ber) if b > 0]
    if valid:
        ex, bx = zip(*valid)
        ax.semilogy(ex, bx, "o--", color=GOLD, lw=1.5,
                    markersize=6, label="Simulation")

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_ylim(1e-6, 1.0)
    ax.legend()
    ax.grid(True, which="both")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. SRRC filter frequency response
# ---------------------------------------------------------------------------

def plot_srrc_response(h: np.ndarray, sps: int,
                       save_path: Optional[str] = None):
    """Plot the SRRC filter magnitude response (normalised frequency)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Square-Root Raised-Cosine Filter Response", fontsize=13, color="#e6edf3")

    nfft = 8192
    H = np.fft.fft(h, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1.0 / sps)
    freqs = np.fft.fftshift(freqs)
    H_mag_dB = 20 * np.log10(np.abs(np.fft.fftshift(H)) + 1e-30)

    ax.plot(freqs, H_mag_dB, color=ACCENT, lw=1.5)
    # Show ±1.5 symbol rates: wide enough to display full passband and
    # first stopband lobe, narrow enough to avoid the confusing aliases
    # that appear at ±(sps/2) = ±4 symbol rates in the oversampled grid.
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Normalised Frequency (×symbol rate)")
    ax.set_ylabel("Magnitude (dB)")
    ax.axhline(-3, color="#8b949e", ls="--", lw=0.8, label="-3 dB")
    ax.legend(fontsize=8)
    ax.grid(True)
    # Caption-style note: the filter is evaluated on the oversampled grid
    # (sps=8), so the full Nyquist range is ±4 symbol rates; only ±1.5
    # is shown here to focus on the passband and transition band.
    ax.text(
        0.99, 0.04,
        f"Shown: ±1.5 sym rates.  Filter: rolloff=0.25, span=10, sps={sps}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#8b949e", style="italic",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
