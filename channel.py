"""
channel.py
----------
RF downlink channel model.

Block ordering updated to match MATLAB/Simulink RF Satellite Link diagram:

  1. Free-space path loss  (Tx antenna gain + FSPL + Rx antenna gain)
  2. Doppler frequency shift
  3. Receiver thermal noise (AWGN)        ← BEFORE LNA (MATLAB order)
  4. LNA gain                             ← AFTER noise  (MATLAB order)
  5. Phase noise                          (optional – colored or white)
  6. I/Q imbalance                        (optional)
  7. DC offset                            (optional – absolute or relative)

Change from original Python
----------------------------
Original Python applied the LNA BEFORE adding thermal noise.  The MATLAB
Simulink block diagram places the Thermal Noise block at the Rx antenna input,
before the LNA.  This file now follows the MATLAB order.

SNR accounting: because noise enters before the LNA, the noise power at the
output of the LNA is G_LNA × P_noise.  The SNR at the LNA output is therefore:

    SNR_out = (G_LNA × P_signal_rx) / (G_LNA × P_noise)
            = P_signal_rx / P_noise

which is the same expression as before — the LNA gain cancels in the ratio.
The SNR reported is therefore physically equivalent; only the block order differs.
"""

import numpy as np
from dataclasses import dataclass

from config import Config
from impairments import (
    apply_path_loss,
    apply_doppler,
    add_awgn_noise,
    add_phase_noise,
    add_colored_phase_noise,
    apply_iq_imbalance,
    add_dc_offset,
    add_dc_offset_relative,
    apply_lna_gain,
)


@dataclass
class ChannelOutput:
    """Container for channel output and diagnostics."""
    signal: np.ndarray   # signal at Rx LNA output (after all channel effects)
    snr_db: float        # estimated SNR (dB) at Rx


def propagate(tx_signal: np.ndarray,
              cfg: Config,
              rng: np.random.Generator,
              override_doppler_hz: float = None,
              override_noise_temp_k: float = None) -> ChannelOutput:
    """
    Pass *tx_signal* through the downlink channel.

    Parameters
    ----------
    tx_signal            : complex 1-D array (HPA output, oversampled)
    cfg                  : simulation Config
    rng                  : seeded random generator
    override_doppler_hz  : overrides cfg.doppler_hz when given
    override_noise_temp_k: overrides cfg.noise_temp_k when given

    Returns
    -------
    ChannelOutput
    """
    doppler_hz    = cfg.doppler_hz    if override_doppler_hz   is None else override_doppler_hz
    noise_temp_k  = cfg.noise_temp_k  if override_noise_temp_k is None else override_noise_temp_k

    sig = tx_signal.copy()

    # ------------------------------------------------------------------
    # 1. Free-space path loss (net: Tx gain – FSPL + Rx gain)
    # ------------------------------------------------------------------
    sig = apply_path_loss(
        sig,
        fspl_db=cfg.free_space_path_loss_db,
        tx_gain_dbi=cfg.tx_antenna_gain_dbi,
        rx_gain_dbi=cfg.rx_antenna_gain_dbi,
    )

    rx_signal_power = float(np.mean(np.abs(sig) ** 2))

    # ------------------------------------------------------------------
    # 2. Doppler frequency offset
    # ------------------------------------------------------------------
    sig = apply_doppler(sig, doppler_hz, cfg.sample_rate_hz)

    # ------------------------------------------------------------------
    # 3. Receiver thermal noise (AWGN)  — BEFORE LNA  (MATLAB order)
    # ------------------------------------------------------------------
    sig = add_awgn_noise(sig, noise_temp_k, cfg.sample_rate_hz, rng)

    # Compute SNR at the Rx antenna (before LNA) for reporting.
    # SNR = P_signal / P_noise.  Since LNA cancels in the ratio (see docstring)
    # we report it here (equivalent to after-LNA SNR).
    from config import K_BOLTZMANN
    noise_power_theoretical = K_BOLTZMANN * noise_temp_k * (cfg.sample_rate_hz / 2.0)
    if noise_temp_k > 0 and rx_signal_power > 0:
        snr_db = 10.0 * np.log10(rx_signal_power / (noise_power_theoretical + 1e-300))
    else:
        snr_db = float('inf')

    # ------------------------------------------------------------------
    # 4. LNA gain  — AFTER noise  (MATLAB order)
    # ------------------------------------------------------------------
    sig = apply_lna_gain(sig, cfg.lna_gain_db)

    # ------------------------------------------------------------------
    # 5. Phase noise
    # ------------------------------------------------------------------
    if cfg.apply_phase_noise:
        if cfg.phase_noise_use_white:
            # Legacy white phase noise
            sig = add_phase_noise(sig, cfg.phase_noise_power_rad2, rng)
        else:
            # Colored 1/f² phase noise – matches MATLAB Phase Noise block
            sig = add_colored_phase_noise(
                sig,
                pn_dbc_hz=cfg.phase_noise_dbc_hz,
                freq_offset_hz=cfg.phase_noise_freq_offset_hz,
                sample_rate=cfg.sample_rate_hz,
                rng=rng,
            )

    # ------------------------------------------------------------------
    # 6. I/Q imbalance
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance:
        sig = apply_iq_imbalance(
            sig,
            cfg.iq_amplitude_imbalance_db,
            cfg.iq_phase_imbalance_deg,
        )

    # ------------------------------------------------------------------
    # 7. DC offset
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset:
        if cfg.dc_offset_mode == "absolute":
            sig = add_dc_offset(sig, cfg.dc_offset_i_abs, cfg.dc_offset_q_abs)
        else:
            # Relative mode (legacy)
            sig = add_dc_offset_relative(sig, cfg.dc_offset_i, cfg.dc_offset_q)

    if cfg.verbose:
        print(f"  [Ch] Rx SNR ≈ {snr_db:.1f} dB | "
              f"Doppler={doppler_hz:.1f} Hz | T_noise={noise_temp_k:.0f} K | "
              f"LNA={cfg.lna_gain_db:.0f} dB (noise-before-LNA order)")

    return ChannelOutput(signal=sig, snr_db=snr_db)
