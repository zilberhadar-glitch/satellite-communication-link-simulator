"""
impairments.py
--------------
All RF impairment models.  Updated to match the MATLAB RF Satellite Link
example more closely.

Changes vs original Python
---------------------------
1. saleh_dpd()        – NEW: analytic inverse-Saleh pre-distorter (LUT-based)
2. add_colored_phase_noise() – NEW: 1/f-shaped phase noise matching the
                               MATLAB Phase Noise block (dBc/Hz at offset)
3. add_dc_offset() updated – supports "absolute" mode (MATLAB default) and
                             legacy "relative" mode
4. LNA ordering in channel.py moved: noise is now added BEFORE the LNA
   (matching MATLAB's Simulink block diagram order)
"""

import numpy as np
from config import K_BOLTZMANN


# ============================================================
# 1.  HPA – Saleh TWTA Memoryless Nonlinearity
# ============================================================

def saleh_hpa(signal: np.ndarray,
              a_a: float, b_a: float,
              a_p: float, b_p: float,
              input_backoff_db: float) -> np.ndarray:
    """
    Saleh model for a Travelling-Wave Tube Amplifier (TWTA).

    AM/AM:  A(r) = a_a * r  /  (1 + b_a * r²)
    AM/PM:  Φ(r) = a_p * r² / (1 + b_p * r²)   [radians]
    """
    r_sat = 1.0 / np.sqrt(b_a)
    ibo_linear = 10 ** (input_backoff_db / 10.0)
    rms_target = r_sat / np.sqrt(ibo_linear)

    rms_in = np.sqrt(np.mean(np.abs(signal) ** 2))
    if rms_in > 0:
        x = signal * (rms_target / rms_in)
    else:
        return signal.copy()

    r = np.abs(x)
    A = a_a * r / (1.0 + b_a * r ** 2)
    phi = a_p * r ** 2 / (1.0 + b_p * r ** 2)
    theta = np.angle(x)
    return A * np.exp(1j * (theta + phi))


def saleh_dpd(signal: np.ndarray,
              a_a: float, b_a: float,
              a_p: float, b_p: float,
              input_backoff_db: float,
              lut_points: int = 1024) -> np.ndarray:
    """
    Digital Pre-Distortion (DPD) for the Saleh TWTA model.

    Implements the analytic inverse of the Saleh AM/AM and AM/PM functions
    using a Look-Up Table (LUT).  The LUT maps the desired output amplitude
    (post-HPA) to the required input amplitude, and the inverse AM/PM gives
    the phase pre-correction to apply.

    MATLAB equivalent: the DPD subsystem in the RF Satellite Link model
    pre-distorts the signal before the HPA so that the combined DPD+HPA
    output is approximately linear.

    Algorithm
    ---------
    1. Build a dense grid of input amplitudes r_in ∈ [0, r_max].
    2. Compute the corresponding HPA outputs: r_out = A(r_in), phi_out = Φ(r_in).
    3. Invert: given a desired amplitude r_des (= current input envelope),
       find r_in via linear interpolation so that A(r_in) ≈ r_des.
    4. Apply the inverse phase shift -Φ(r_in) so the HPA output phase is
       aligned with the un-distorted input phase.

    Parameters
    ----------
    signal           : complex 1-D array (normalised to match the HPA operating point)
    a_a, b_a         : Saleh AM/AM coefficients
    a_p, b_p         : Saleh AM/PM coefficients
    input_backoff_db : IBO used for the downstream HPA (must match)
    lut_points       : number of LUT entries (finer = more accurate)

    Returns
    -------
    pre_distorted : complex 1-D array ready to be passed through saleh_hpa()
    """
    # --- 1. Scale input to HPA operating point (same as saleh_hpa) ---
    r_sat = 1.0 / np.sqrt(b_a)
    ibo_linear = 10 ** (input_backoff_db / 10.0)
    rms_target = r_sat / np.sqrt(ibo_linear)
    rms_in = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    if rms_in < 1e-30:
        return signal.copy()
    x = signal * (rms_target / rms_in)

    # --- 2. Build forward LUT:  r_in → r_out, phi_out ---
    # Use 2× the saturation amplitude as the LUT range so even the most
    # compressed operating points are covered.
    r_max = 2.0 * r_sat
    r_in_lut = np.linspace(0.0, r_max, lut_points)
    r_out_lut = a_a * r_in_lut / (1.0 + b_a * r_in_lut ** 2)     # AM/AM output
    phi_lut   = a_p * r_in_lut ** 2 / (1.0 + b_p * r_in_lut ** 2) # AM/PM phase

    # The AM/AM curve rises to r_sat then falls.  The LUT is therefore only
    # monotone on [0, r_sat].  We restrict inversion to this region:
    i_sat = np.argmax(r_out_lut)      # index of saturation peak
    r_in_mono  = r_in_lut[:i_sat + 1]
    r_out_mono = r_out_lut[:i_sat + 1]
    phi_mono   = phi_lut[:i_sat + 1]

    # --- 3. Invert: desired amplitude r_des → required input r_pre ---
    r_des = np.abs(x)                          # desired amplitude (= clean input)

    # Clip desired amplitude to the monotone range (beyond r_sat the HPA
    # always compresses; we clamp to the peak output)
    r_des_clipped = np.clip(r_des, 0.0, r_out_mono[-1])

    # Linear interpolation of the inverse mapping
    r_pre  = np.interp(r_des_clipped, r_out_mono, r_in_mono)
    phi_pre = np.interp(r_pre, r_in_mono, phi_mono)

    # --- 4. Reconstruct pre-distorted complex signal ---
    theta = np.angle(x)
    pre_distorted_scaled = r_pre * np.exp(1j * (theta - phi_pre))

    # Scale back from the HPA operating domain to the original signal domain
    out = pre_distorted_scaled * (rms_in / rms_target)
    return out


def hpa_am_am_curve(a_a: float, b_a: float, n_points: int = 200):
    """Return (r_in, r_out) arrays for plotting the AM/AM characteristic."""
    r_sat = 1.0 / np.sqrt(b_a)
    r_in = np.linspace(0, 2 * r_sat, n_points)
    r_out = a_a * r_in / (1.0 + b_a * r_in ** 2)
    return r_in, r_out


# ============================================================
# 2.  Free-Space Path Loss
# ============================================================

def apply_path_loss(signal: np.ndarray,
                    fspl_db: float,
                    tx_gain_dbi: float,
                    rx_gain_dbi: float) -> np.ndarray:
    """
    Scale signal by Tx gain – FSPL + Rx gain.
    Net voltage gain = 10^( (Gt - FSPL + Gr) / 20 ).
    """
    net_db = tx_gain_dbi - fspl_db + rx_gain_dbi
    voltage_gain = 10.0 ** (net_db / 20.0)
    return signal * voltage_gain


# ============================================================
# 3.  Doppler Frequency Offset
# ============================================================

def apply_doppler(signal: np.ndarray,
                  doppler_hz: float,
                  sample_rate: float) -> np.ndarray:
    """Complex exponential Doppler shift."""
    if doppler_hz == 0.0:
        return signal.copy()
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * doppler_hz * t)


# ============================================================
# 4.  AWGN – Receiver Thermal Noise
# ============================================================

def add_awgn_noise(signal: np.ndarray,
                   noise_temp_k: float,
                   sample_rate: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Add complex AWGN whose power follows P_noise = k_B * T * B.

    MATLAB block order: noise is added BEFORE the LNA (see channel.py).
    The LNA then amplifies both signal and noise together.
    """
    if noise_temp_k <= 0:
        return signal.copy()

    noise_power = K_BOLTZMANN * noise_temp_k * (sample_rate / 2.0)
    if np.mean(np.abs(signal) ** 2) <= 0:
        return signal.copy()

    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal))
                     + 1j * rng.standard_normal(len(signal)))
    return signal + noise


# ============================================================
# 5a. Phase Noise – Colored (MATLAB Phase Noise block equivalent)
# ============================================================

def add_colored_phase_noise(signal: np.ndarray,
                             pn_dbc_hz: float,
                             freq_offset_hz: float,
                             sample_rate: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Add colored (1/f²) phase noise shaped to match a specified single-sideband
    phase noise level (dBc/Hz) at a given frequency offset.

    MATLAB equivalent: comm.PhaseNoise (or the Phase Noise block in Simulink)
    which specifies noise in dBc/Hz at a reference offset.

    Model
    -----
    The phase noise PSD is approximated as:

        S_phi(f) = L_0 * (f_0 / f)^2         [rad²/Hz]

    where L_0 = 10^(pn_dbc_hz/10) is the single-sideband level at f_0.

    This is a 1/f² Lorentzian model, the dominant term for free-running
    oscillators and the standard simplified model for MATLAB's Phase Noise block.

    Implementation: shape white noise in the frequency domain, then apply
    as a phase modulation in the time domain.

    Comparison to MATLAB
    --------------------
    MATLAB's comm.PhaseNoise block uses an interpolated PSD from a user-supplied
    table.  This Python version uses the single-point 1/f² approximation.
    For a single (offset, level) specification the two approaches agree at that
    offset and diverge at other frequencies.  This is an acceptable approximation
    for coursework purposes; exact reproduction would require the full PSD table
    from the MATLAB model.

    Parameters
    ----------
    signal         : complex 1-D array
    pn_dbc_hz      : single-sideband phase noise level at freq_offset_hz  (dBc/Hz)
    freq_offset_hz : reference offset frequency (Hz)
    sample_rate    : sample rate of the signal (same units as freq_offset_hz)
    rng            : seeded random generator
    """
    if pn_dbc_hz >= 0:
        return signal.copy()   # non-negative dBc is unphysical; skip

    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)   # [0, fs/2]
    freqs[0] = freqs[1]   # avoid divide-by-zero at DC; will be zeroed anyway

    # L_0 at the reference offset
    L0 = 10.0 ** (pn_dbc_hz / 10.0)   # linear units (rad²/Hz)

    # 1/f² PSD: S(f) = L0 * (f0/f)^2
    S_phi = L0 * (freq_offset_hz / freqs) ** 2   # [rad²/Hz]
    S_phi[0] = 0.0   # no DC phase drift

    # Per-bin amplitude (one-sided FFT → sqrt(2) for correct total power)
    amp = np.sqrt(S_phi * sample_rate / N) * np.sqrt(2.0)

    # Generate white noise in frequency domain, then shape
    noise_f = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    phase_f = amp * noise_f

    # Convert to time-domain phase
    phase_t = np.fft.irfft(phase_f, n=N)   # real-valued phase noise

    return signal * np.exp(1j * phase_t)


# ============================================================
# 5b. Phase Noise – White (legacy simplified model)
# ============================================================

def add_phase_noise(signal: np.ndarray,
                    phase_noise_var: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Add sample-by-sample additive white phase noise (legacy model).
    phi[n] ~ N(0, phase_noise_var)  (radians)
    """
    if phase_noise_var <= 0:
        return signal.copy()
    phi = rng.normal(0.0, np.sqrt(phase_noise_var), len(signal))
    return signal * np.exp(1j * phi)


# ============================================================
# 6.  I/Q Imbalance
# ============================================================

def apply_iq_imbalance(signal: np.ndarray,
                       amplitude_imbalance_db: float,
                       phase_imbalance_deg: float) -> np.ndarray:
    """
    Model I/Q mixer imbalance.

        I_out = I_in
        Q_out = (1 + ε) * Q_in  +  I_in * sin(Δφ)

    MATLAB default values: amplitude_imbalance_db=3 dB, phase_imbalance_deg=20°.
    Setting one parameter to 0 gives amplitude-only or phase-only imbalance.
    """
    if amplitude_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
        return signal.copy()

    eps  = 10 ** (amplitude_imbalance_db / 20.0) - 1.0
    dphi = np.deg2rad(phase_imbalance_deg)

    I_out = signal.real
    Q_out = (1.0 + eps) * signal.imag + signal.real * np.sin(dphi)
    return I_out + 1j * Q_out


# ============================================================
# 7.  DC Offset
# ============================================================

def add_dc_offset(signal: np.ndarray,
                  dc_i: float,
                  dc_q: float) -> np.ndarray:
    """Add a complex DC offset to the signal (absolute value)."""
    return signal + (dc_i + 1j * dc_q)


def add_dc_offset_relative(signal: np.ndarray,
                            frac_i: float,
                            frac_q: float) -> np.ndarray:
    """
    Add a DC offset expressed as a fraction of signal RMS.
    Legacy mode for relative offsets.
    """
    sig_rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    return signal + ((frac_i + 1j * frac_q) * sig_rms)


# ============================================================
# 8.  LNA Gain
# ============================================================

def apply_lna_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a linear voltage gain (dB) to the signal."""
    return signal * 10.0 ** (gain_db / 20.0)
