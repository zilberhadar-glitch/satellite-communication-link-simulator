"""
receiver.py
-----------
Complete satellite downlink receiver chain:

  1. SRRC matched filter + downsample
  2. DC offset correction          (optional)
  3. AGC (Automatic Gain Control)  (optional)
  4. I/Q imbalance compensation    (optional)
  5. Doppler / carrier-frequency correction (optional)
  6. 16-QAM hard-decision demodulation
  7. BER / SER calculation

All compensators use simple closed-form estimates rather than adaptive
algorithms so the behaviour is deterministic and easy to understand.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from filters import rx_filter, filter_delay
from modulation import symbols_to_bits, symbol_error_rate


# ---------------------------------------------------------------------------
# Compensator helpers
# ---------------------------------------------------------------------------

def _dc_correction(symbols: np.ndarray) -> np.ndarray:
    """Remove estimated DC bias (sample mean)."""
    return symbols - np.mean(symbols)


def _agc(symbols: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """Scale symbols so that their average power equals *target_power*."""
    pwr = np.mean(np.abs(symbols) ** 2)
    if pwr > 0:
        return symbols * np.sqrt(target_power / pwr)
    return symbols


def _iq_correction_exact(symbols: np.ndarray) -> np.ndarray:
    """
    Model-matched I/Q imbalance correction.

    The impairment model in impairments.apply_iq_imbalance() is:
        I_out = I_in
        Q_out = (1 + eps) * Q_in  +  I_in * sin(dphi)

    This is algebraically equivalent to:
        y = A * x  +  B * conj(x)
    with
        A = (2 + eps + j*sin(dphi)) / 2
        B = (  -eps + j*sin(dphi)) / 2

    Parameter estimation via cross-moments
    ----------------------------------------
    Because I_out = I_in, the I rail is untouched. This gives:

        E[Re(y) * Im(y)] = E[I_in * Q_out]
                         = E[I * ((1+eps)*Q + I*sin)]
                         = (1+eps)*E[I*Q]  +  sin*E[I^2]
                         = 0  +  sin * 0.5        (balanced 16-QAM: E[I*Q]=0)
        => sin(dphi) = 2 * E[Re(y) * Im(y)]

        E[Im(y)^2] = E[((1+eps)*Q + I*sin)^2]
                   = (1+eps)^2 * 0.5  +  sin^2 * 0.5
        => eps = sqrt(2*E[Im(y)^2] - sin(dphi)^2) - 1

    Correction (exact inverse)
    --------------------------
        x = (conj(A) * y  -  B * conj(y)) / (|A|^2 - |B|^2)

    Important: this function must be called BEFORE AGC.
    AGC rescales amplitude uniformly, which alters E[Im^2] and would
    break the eps estimate. The sin estimate is AGC-invariant in theory
    but the eps estimate is not, so always correct before AGC.

    Detection guard
    ---------------
    If both |sin_est| < 0.005 and |eps_est| < 0.001 the signal
    appears balanced; skip correction to avoid injecting finite-sample
    noise into a clean signal.
    """
    # Normalise by average symbol power so estimates are scale-invariant.
    # After the matched filter, symbols are at physical signal amplitude
    # (~10^-7 after path loss), not normalised to 1. Without this division
    # the cross-moments are on the order of 10^-15 and eps_est collapses to -1.
    pwr = float(np.mean(np.abs(symbols) ** 2))
    if pwr < 1e-30:
        return symbols

    # Estimate sin(dphi) from the normalised I-Q cross-moment
    sin_est = 2.0 * float(np.mean(symbols.real * symbols.imag)) / pwr

    # Estimate eps from the normalised Q-rail power
    eimq2_n = float(np.mean(symbols.imag ** 2)) / pwr
    radicand = 2.0 * eimq2_n - sin_est ** 2
    if radicand < 0.0:
        return symbols      # numerical issue; leave symbols unchanged

    eps_est = float(np.sqrt(radicand)) - 1.0

    # Detection guard — skip when both parameters are below noise floor
    if abs(sin_est) < 0.005 and abs(eps_est) < 0.001:
        return symbols

    # Reconstruct A and B from the estimated parameters
    A = (2.0 + eps_est + 1j * sin_est) / 2.0
    B = (      -eps_est + 1j * sin_est) / 2.0

    denom = abs(A) ** 2 - abs(B) ** 2
    if abs(denom) < 1e-10:
        return symbols

    return (np.conj(A) * symbols - B * np.conj(symbols)) / denom


# Public alias — nothing outside this file needs to change.
_iq_imbalance_correction = _iq_correction_exact


def iq_correct_ideal(symbols: np.ndarray,
                     amp_db: float,
                     phase_deg: float) -> np.ndarray:
    """
    Ideal I/Q imbalance correction using the exact known impairment parameters.

    This bypasses estimation entirely and applies the mathematically exact
    inverse of the impairment model in impairments.apply_iq_imbalance().

    Use this as an upper bound / validation mode to confirm that the
    correction formula is correct independent of estimator quality.

    Parameters
    ----------
    symbols   : received symbols (after matched filter, before AGC)
    amp_db    : the amplitude imbalance that was applied (dB) — must be known
    phase_deg : the phase imbalance that was applied (degrees) — must be known

    Returns
    -------
    Corrected symbols at the same scale as the input.
    """
    if amp_db == 0.0 and phase_deg == 0.0:
        return symbols

    eps  = 10.0 ** (amp_db / 20.0) - 1.0
    dphi = np.deg2rad(phase_deg)

    A = (2.0 + eps + 1j * np.sin(dphi)) / 2.0
    B = (      -eps + 1j * np.sin(dphi)) / 2.0

    denom = abs(A) ** 2 - abs(B) ** 2
    if abs(denom) < 1e-10:
        return symbols

    return (np.conj(A) * symbols - B * np.conj(symbols)) / denom


def _cfo_estimate_symbol_rate(symbols: np.ndarray,
                               fs_sym: float = 1.0,
                               skip: int = 0,
                               zp_factor: int = 4) -> float:
    """
    Estimate the carrier frequency offset (CFO) from symbol-rate samples.

    Uses the 4th-power non-data-aided (NDA) method on the downsampled
    (1 sample/symbol) sequence, with zero-padding to achieve sub-bin
    frequency resolution.

    Why symbol-rate, not oversampled?
    ----------------------------------
    The previous implementation ran on the oversampled waveform before
    matched filtering.  This has two problems:

    1. The SRRC filter has a passband of ±0.5/sps of the oversampled rate,
       so Doppler offsets larger than that shift the signal outside the filter
       band.  On the oversampled FFT the 4th-power tone lands outside the
       meaningful band.
    2. The SRRC pulse shape spreads energy, so the oversampled 4th-power
       spectrum has no clean narrow tone even for large Doppler.

    At symbol rate the signal is a flat-spectrum QAM sequence and the
    4th-power tone is a clean spike at 4·f_CFO within [−2, +2] Hz.

    Why zero-padding?
    -----------------
    Without zero-padding the FFT bin spacing is fs_sym/N = 1/10000 = 0.1 mHz,
    which is sufficient.  However, the raw FFT peak sits on the nearest bin,
    causing a quantisation error of up to fs_sym/(2·N) per estimate.  Over
    N symbols this residual error accumulates as a phase ramp of up to
    π radians, causing a BER floor.  4× zero-padding reduces the grid
    spacing by 4× and the residual phase drift by 4×, enough to eliminate
    the BER floor for offsets up to ~5% of symbol rate.

    Parameters
    ----------
    symbols   : complex 1-D array at 1 sample/symbol (after matched filter)
    fs_sym    : symbol rate in the simulation's normalised units (default 1.0)
    skip      : number of leading symbols to discard (filter settling transient)
    zp_factor : zero-padding multiplier for finer frequency grid (default 4)

    Returns
    -------
    estimated CFO in the same units as fs_sym
    """
    s = symbols[skip:]
    s4 = s ** 4
    nfft = len(s4) * zp_factor
    spectrum = np.fft.fft(s4, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs_sym)
    peak_idx = np.argmax(np.abs(spectrum))
    return float(freqs[peak_idx] / 4.0)


def _phase_estimate_data_aided(symbols: np.ndarray,
                                ref_symbols: np.ndarray,
                                skip: int = 0,
                                n_pilot: int = 200) -> float:
    """
    Estimate the residual constant phase offset using known reference symbols.

    After CFO correction there remains a constant (or slowly varying) phase
    offset phi_0 that depends on the signal's initial phase when it entered
    the channel.  This function estimates phi_0 from a short block of known
    symbols (pilots or known preamble data).

    In simulation, the transmitted symbols are always available, so this
    implements a 'data-aided' or 'pilot-aided' estimator.  In hardware this
    would be done with a known preamble or with decision-directed tracking.

    Parameters
    ----------
    symbols     : received symbols after CFO correction
    ref_symbols : ideal transmitted symbols (used as the pilot reference)
    skip        : leading symbols to skip (filter settling)
    n_pilot     : number of symbols to use for the average

    Returns
    -------
    estimated phase offset in radians
    """
    i0 = skip
    i1 = min(i0 + n_pilot, len(symbols), len(ref_symbols))
    if i1 <= i0:
        return 0.0
    return float(np.angle(np.mean(symbols[i0:i1] * np.conj(ref_symbols[i0:i1]))))


def _doppler_correction(signal: np.ndarray,
                        sample_rate: float,
                        carrier_freq_hz: float,
                        coarse_freq_est_hz: float = None) -> tuple:
    """
    Carrier-frequency-offset (CFO) correction on the OVERSAMPLED signal.

    This function handles the case where the true (or externally provided)
    CFO is known.  When coarse_freq_est_hz is None, it returns the signal
    unchanged and estimated_offset=0.0; the actual blind estimation is
    deferred to after matched filtering (see receive() step 2b).

    Parameters
    ----------
    signal              : oversampled complex signal from the channel
    sample_rate         : oversampled sample rate (sps × symbol rate)
    carrier_freq_hz     : RF carrier (unused here, kept for API compatibility)
    coarse_freq_est_hz  : if given, apply this correction immediately
                          (ideal mode — the true CFO is passed in)

    Returns
    -------
    (corrected_signal, estimated_offset_hz)
    """
    if coarse_freq_est_hz is None:
        # Blind estimation is deferred to after matched filtering.
        # Return the signal unmodified; receiver() will call
        # _cfo_estimate_symbol_rate() on the downsampled symbols.
        return signal.copy(), 0.0

    # Ideal (or externally supplied) correction on the oversampled signal.
    t = np.arange(len(signal)) / sample_rate
    corrected = signal * np.exp(-1j * 2 * np.pi * coarse_freq_est_hz * t)
    return corrected, float(coarse_freq_est_hz)


# ---------------------------------------------------------------------------
# Main receiver function
# ---------------------------------------------------------------------------

@dataclass
class RxResult:
    """Container for receiver outputs."""
    symbols: np.ndarray           # demodulated symbols (post-matched-filter)
    rx_bits: np.ndarray           # hard-decision bits
    ber: float
    ser: float
    n_bit_errors: int
    estimated_doppler_hz: float = 0.0


def receive(rx_signal: np.ndarray,
            tx_bits: np.ndarray,
            cfg: Config,
            override_doppler_hz: float = None,
            ref_symbols: np.ndarray = None) -> 'RxResult':
    """
    Run the full receiver chain.

    Parameters
    ----------
    rx_signal          : complex oversampled signal from the channel
    tx_bits            : reference bit stream (for BER computation)
    cfg                : simulation Config
    override_doppler_hz: if given, apply this as the known CFO on the
                         oversampled signal before matched filtering
                         (ideal mode — does NOT use the blind estimator)
    ref_symbols        : ideal transmitted symbols used for data-aided
                         phase correction after blind CFO estimation.
                         If None the function derives them from tx_bits.

    Returns
    -------
    RxResult
    """
    from modulation import bits_to_symbols as _b2s
    sig = rx_signal.copy()

    # ------------------------------------------------------------------
    # 1. Ideal CFO correction on oversampled signal (only when the true
    #    Doppler is explicitly provided; this is the "ideal" test mode).
    #    Blind estimation is deferred to step 2b, after matched filtering,
    #    where the estimator has access to clean symbol-rate samples.
    # ------------------------------------------------------------------
    ideal_cfo_applied = False
    if cfg.apply_doppler_correction and override_doppler_hz is not None:
        sig, _ = _doppler_correction(
            sig,
            sample_rate=cfg.sample_rate_hz,
            carrier_freq_hz=cfg.carrier_freq_hz,
            coarse_freq_est_hz=override_doppler_hz,
        )
        ideal_cfo_applied = True
        if cfg.verbose:
            print(f"  [Rx] CFO correction (ideal): {override_doppler_hz:.6f} Hz")

    # ------------------------------------------------------------------
    # 2. SRRC matched filter + downsample to 1 sample/symbol
    # ------------------------------------------------------------------
    delay = filter_delay(cfg.span, cfg.samples_per_symbol)
    symbols = rx_filter(sig,
                        cfg.srrc_h if hasattr(cfg, 'srrc_h') else _get_srrc_h(cfg),
                        cfg.samples_per_symbol, delay)

    # ------------------------------------------------------------------
    # 2b. Blind CFO estimation and correction on symbol-rate data.
    #     This runs only when Doppler correction is requested but no true
    #     value was supplied.  The estimator uses the 4th-power method
    #     with zero-padding on the downsampled symbols, which gives
    #     sub-bin frequency resolution and avoids the SRRC spectral
    #     distortion that occurs on the oversampled waveform.
    # ------------------------------------------------------------------
    estimated_doppler = 0.0
    if cfg.apply_doppler_correction and not ideal_cfo_applied:
        n_settle = cfg.span   # discard filter settling transient
        estimated_doppler = _cfo_estimate_symbol_rate(
            symbols,
            fs_sym=1.0,          # normalised symbol rate
            skip=n_settle,
            zp_factor=4,
        )
        t_sym = np.arange(len(symbols))
        symbols = symbols * np.exp(-1j * 2 * np.pi * estimated_doppler * t_sym)
        if cfg.verbose:
            print(f"  [Rx] CFO correction (blind): estimated={estimated_doppler:.6f} Hz")

    elif ideal_cfo_applied:
        estimated_doppler = float(override_doppler_hz)

    # ------------------------------------------------------------------
    # 2c. Data-aided residual phase correction.
    #     After CFO removal a constant phase offset phi_0 remains (it
    #     depends on where in the symbol period the channel's Doppler ramp
    #     started).  We estimate phi_0 from a short block of known symbols,
    #     skipping the filter settling period.
    #     This step runs whenever Doppler correction is active.
    # ------------------------------------------------------------------
    if cfg.apply_doppler_correction:
        if ref_symbols is None:
            ref_symbols = _b2s(tx_bits, cfg.modulation_order)
        n_settle = cfg.span
        phase_offset = _phase_estimate_data_aided(
            symbols, ref_symbols,
            skip=n_settle,
            n_pilot=200,
        )
        symbols = symbols * np.exp(-1j * phase_offset)
        if cfg.verbose:
            print(f"  [Rx] Phase correction (data-aided): {np.degrees(phase_offset):.2f} deg")

    # ------------------------------------------------------------------
    # 3. DC offset correction
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset and cfg.apply_dc_correction:
        symbols = _dc_correction(symbols)

    # ------------------------------------------------------------------
    # 4. I/Q imbalance compensation  (must run BEFORE AGC)
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance and cfg.apply_iq_correction:
        symbols = _iq_imbalance_correction(symbols)

    # ------------------------------------------------------------------
    # 5. AGC
    # ------------------------------------------------------------------
    if cfg.apply_agc:
        symbols = _agc(symbols, target_power=cfg.agc_target_power)

    # ------------------------------------------------------------------
    # 6. Hard-decision 16-QAM demodulation → bits
    # ------------------------------------------------------------------
    rx_bits = symbols_to_bits(symbols, order=cfg.modulation_order)

    # ------------------------------------------------------------------
    # 7. BER / SER  (align lengths)
    # ------------------------------------------------------------------
    ber, ser, n_errors = symbol_error_rate(tx_bits, rx_bits, cfg.bits_per_symbol)

    if cfg.verbose:
        print(f"  [Rx] BER = {ber:.2e}  ({n_errors}/{min(len(tx_bits),len(rx_bits))} bit errors)")

    return RxResult(
        symbols=symbols,
        rx_bits=rx_bits,
        ber=ber,
        ser=ser,
        n_bit_errors=n_errors,
        estimated_doppler_hz=estimated_doppler,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_srrc_h(cfg: Config) -> np.ndarray:
    """Compute SRRC coefficients from config (avoid circular import)."""
    from filters import srrc_coeffs
    return srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)


def attach_srrc_h(cfg: Config) -> None:
    """Compute and cache SRRC coefficients on the config object."""
    cfg.srrc_h = _get_srrc_h(cfg)
