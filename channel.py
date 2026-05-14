"""
channel.py
----------
RF downlink channel model.

Applies, in order:
  1. Free-space path loss  (Tx antenna gain + FSPL + Rx antenna gain)
  2. Doppler frequency shift
  3. Receiver thermal noise (AWGN scaled by noise temperature)
  4. Phase noise              (optional)
  5. I/Q imbalance            (optional)
  6. DC offset                (optional)
  7. LNA gain                 (Rx front-end amplifier)
"""

import numpy as np
from dataclasses import dataclass

from config import Config
from impairments import (
    apply_path_loss,
    apply_doppler,
    add_awgn_noise,
    add_phase_noise,
    apply_iq_imbalance,
    add_dc_offset,
    apply_lna_gain,
)


@dataclass
class ChannelOutput:
    """Container for channel output and diagnostics."""
    signal: np.ndarray       # signal at Rx input (after all channel effects)
    snr_db: float            # estimated SNR (dB) at Rx input


def propagate(tx_signal: np.ndarray,
              cfg: Config,
              rng: np.random.Generator,
              override_doppler_hz: float = None,
              override_noise_temp_k: float = None) -> ChannelOutput:
    """
    Pass *tx_signal* through the downlink channel.

    Parameters
    ----------
    tx_signal            : complex 1-D array (output of the HPA, oversampled)
    cfg                  : simulation Config
    rng                  : seeded random generator
    override_doppler_hz  : if given, overrides cfg.doppler_hz
    override_noise_temp_k: if given, overrides cfg.noise_temp_k

    Returns
    -------
    ChannelOutput
    """
    doppler_hz = override_doppler_hz if override_doppler_hz is not None else cfg.doppler_hz
    noise_temp_k = override_noise_temp_k if override_noise_temp_k is not None else cfg.noise_temp_k

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

    rx_signal_power = np.mean(np.abs(sig) ** 2)

    # ------------------------------------------------------------------
    # 2. Doppler frequency offset
    # ------------------------------------------------------------------
    sig = apply_doppler(sig, doppler_hz, cfg.sample_rate_hz)

    # ------------------------------------------------------------------
    # 3. LNA gain (amplify before noise for SNR accounting)
    # ------------------------------------------------------------------
    sig = apply_lna_gain(sig, cfg.lna_gain_db)

    # ------------------------------------------------------------------
    # 4. Thermal noise (AWGN)
    # ------------------------------------------------------------------
    sig = add_awgn_noise(sig, noise_temp_k, cfg.sample_rate_hz, rng)

    # Estimate SNR at this point
    noise_power_est = max(np.mean(np.abs(sig) ** 2) - np.mean(np.abs(sig * 0) ** 2), 1e-30)
    signal_power_after_lna = rx_signal_power * 10 ** (cfg.lna_gain_db / 10.0)
    from config import K_BOLTZMANN
    noise_power_theoretical = K_BOLTZMANN * noise_temp_k * cfg.sample_rate_hz / 2.0
    if noise_temp_k > 0:
        snr_db = 10 * np.log10(signal_power_after_lna / (noise_power_theoretical + 1e-300))
    else:
        snr_db = float('inf')

    # ------------------------------------------------------------------
    # 5. Phase noise (optional)
    # ------------------------------------------------------------------
    if cfg.apply_phase_noise:
        sig = add_phase_noise(sig, cfg.phase_noise_power_rad2, rng)

    # ------------------------------------------------------------------
    # 6. I/Q imbalance (optional)
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance:
        sig = apply_iq_imbalance(
            sig,
            cfg.iq_amplitude_imbalance_db,
            cfg.iq_phase_imbalance_deg,
        )

    # ------------------------------------------------------------------
    # 7. DC offset (optional)
    # DC values are expressed as fractions of the RMS signal amplitude
    # so the offset is meaningful regardless of the physical power scale.
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset:
        sig_rms = float(np.sqrt(np.mean(np.abs(sig) ** 2)))
        sig = add_dc_offset(sig,
                            cfg.dc_offset_i * sig_rms,
                            cfg.dc_offset_q * sig_rms)

    if cfg.verbose:
        print(f"  [Ch] Rx SNR ≈ {snr_db:.1f} dB | "
              f"Doppler={doppler_hz:.1f} Hz | T_noise={noise_temp_k:.0f} K")

    return ChannelOutput(signal=sig, snr_db=snr_db)
