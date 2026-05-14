"""
main.py
-------
Orchestrator and experiment runner for the RF Satellite Link simulation.

Scenarios simulated:
  1. Clean channel            (no noise, no impairments)
  2. Thermal noise only       (T=290 K)
  3. Noise – low temperature  (T=20 K)
  4. Noise – high temperature (T=500 K)
  5. Doppler (3 Hz) without correction
  6. Doppler (3 Hz) with correction
  7. HPA nonlinearity: IBO = 30 dB (near linear)
  8. HPA nonlinearity: IBO = 7 dB  (moderate compression)
  9. HPA nonlinearity: IBO = 1 dB  (heavy clipping)
 10. I/Q imbalance without correction
 11. I/Q imbalance with correction
 12. Phase noise
 13. DC offset with correction

Usage:
    python main.py
"""

import copy
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file output
import matplotlib.pyplot as plt

from config import Config
from transmitter import transmit, TxSignals
from channel import propagate, ChannelOutput
from receiver import receive, attach_srrc_h
from metrics import (
    ScenarioResult, compute_ebn0_db, compute_evm, compute_papr_db
)
from filters import srrc_coeffs, filter_delay
from modulation import bits_to_symbols
import plots as P


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "output_figures")
os.makedirs(OUT_DIR, exist_ok=True)


def fig_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)


# ===========================================================================
# Core simulation function
# ===========================================================================

def run_simulation(cfg: Config,
                   rng: np.random.Generator,
                   scenario_name: str,
                   override_doppler_hz: float = None,
                   override_noise_temp_k: float = None,
                   custom_backoff_db: float = None) -> ScenarioResult:
    """
    Run one complete Tx → Channel → Rx simulation.

    Returns a ScenarioResult with BER, SNR, EVM, PAPR.
    """
    # Cache the SRRC filter coefficients on cfg so the receiver can reuse them
    attach_srrc_h(cfg)

    # ----- Transmitter -----
    tx = transmit(cfg, rng, custom_backoff_db=custom_backoff_db)

    # ----- Channel -----
    ch = propagate(tx.after_hpa, cfg, rng,
                   override_doppler_hz=override_doppler_hz,
                   override_noise_temp_k=override_noise_temp_k)

    # ----- Receiver -----
    rx = receive(ch.signal, tx.bits, cfg,
                 override_doppler_hz=(override_doppler_hz
                                      if cfg.apply_doppler_correction else None))

    # ----- Metrics -----
    result = ScenarioResult(scenario_name)
    result.ber = rx.ber
    result.ser = rx.ser
    result.n_errors = rx.n_bit_errors
    result.snr_db = ch.snr_db

    nt = override_noise_temp_k if override_noise_temp_k is not None else cfg.noise_temp_k
    result.ebn0_db = compute_ebn0_db(cfg, noise_temp_k=nt)

    # EVM: compare Rx symbols to ideal Tx symbols (re-generate from bits)
    tx_ideal_syms = bits_to_symbols(tx.bits, cfg.modulation_order)
    result.evm_pct = compute_evm(tx_ideal_syms, rx.symbols)
    result.papr_db = compute_papr_db(tx.after_hpa)

    return result, tx, ch, rx


# ===========================================================================
# Experiment: sweep Eb/N0 for theoretical vs simulated BER
# ===========================================================================

def run_ebn0_sweep(cfg_base: Config,
                   ebn0_db_values: list,
                   rng_seed: int = 0) -> tuple:
    """
    Simulate BER at several Eb/N0 values by injecting AWGN directly
    at the correct level (bypassing the physical path-loss scaling).

    This generates the simulation BER curve to compare against theory.
    Returns (ebn0_list, ber_list).
    """
    from filters import srrc_coeffs, tx_filter, rx_filter, filter_delay
    from modulation import bits_to_symbols, symbols_to_bits, symbol_error_rate

    cfg = copy.deepcopy(cfg_base)
    cfg.verbose = False
    cfg.apply_doppler_correction = False
    cfg.apply_phase_noise = False
    cfg.apply_iq_imbalance = False
    cfg.apply_dc_offset = False
    cfg.apply_agc = True
    cfg.hpa_input_backoff_db = 30.0   # near-linear HPA

    h = srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)
    delay = filter_delay(cfg.span, cfg.samples_per_symbol)

    ebn0_list, ber_list = [], []

    for ebn0_db in ebn0_db_values:
        rng = np.random.default_rng(rng_seed)

        # Generate and Tx-filter symbols
        bits = rng.integers(0, 2, size=cfg.num_symbols * cfg.bits_per_symbol)
        symbols = bits_to_symbols(bits, cfg.modulation_order)
        tx_sig = tx_filter(symbols, h, cfg.samples_per_symbol)
        tx_sig /= np.sqrt(np.mean(np.abs(tx_sig) ** 2))   # unit power

        # Convert Eb/N0 → Es/N0 → SNR per sample
        # Es/N0 = Eb/N0 * bits_per_symbol
        # SNR_sample = Es/N0 / samples_per_symbol   (for raised-cosine Nyquist)
        ebn0_lin = 10 ** (ebn0_db / 10.0)
        esn0_lin = ebn0_lin * cfg.bits_per_symbol
        snr_per_sample = esn0_lin / cfg.samples_per_symbol

        # sigma² per complex sample = 1 / (2 * SNR_sample)
        sigma = np.sqrt(1.0 / (2.0 * snr_per_sample))
        noise = sigma * (rng.standard_normal(len(tx_sig))
                         + 1j * rng.standard_normal(len(tx_sig)))
        rx_sig = tx_sig + noise

        # Rx filter + downsample
        rx_syms = rx_filter(rx_sig, h, cfg.samples_per_symbol, delay)

        # AGC
        pwr = np.mean(np.abs(rx_syms) ** 2)
        if pwr > 0:
            rx_syms /= np.sqrt(pwr)

        # Demodulate
        rx_bits = symbols_to_bits(rx_syms[:cfg.num_symbols], cfg.modulation_order)
        ber, _, _ = symbol_error_rate(bits, rx_bits, cfg.bits_per_symbol)

        ebn0_list.append(ebn0_db)
        ber_list.append(ber)
        print(f"  Eb/N0={ebn0_db:>5.1f} dB  BER={ber:.3e}")

    return ebn0_list, ber_list


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 65)
    print("  RF Satellite Link Simulation – Python Implementation")
    print("  Replicating MATLAB/Simulink RF Satellite Link Example")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Base configuration
    # ------------------------------------------------------------------
    base_cfg = Config(
        modulation_order=16,
        carrier_freq_hz=4.0e9,
        sat_altitude_km=35_600.0,
        tx_antenna_diameter_m=0.4,
        rx_antenna_diameter_m=0.4,
        lna_gain_db=30.0,
        noise_temp_k=290.0,
        hpa_input_backoff_db=7.0,
        doppler_hz=3.0,
        rolloff=0.25,
        span=10,
        samples_per_symbol=8,
        num_symbols=10_000,
        random_seed=42,
        apply_doppler_correction=False,
        apply_phase_noise=False,
        apply_iq_imbalance=False,
        apply_dc_offset=False,
        apply_agc=True,
        verbose=True,
    )
    print(base_cfg.summary())

    scenarios: list[ScenarioResult] = []

    # ==================================================================
    # SCENARIO 1 – Clean channel (no noise, no impairments)
    # ==================================================================
    print("\n--- Scenario 1: Clean channel ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, tx_clean, ch_clean, rx_clean = run_simulation(
        cfg, rng, "1. Clean (no noise)", override_noise_temp_k=0.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 2 – Thermal noise T=290 K
    # ==================================================================
    print("\n--- Scenario 2: Thermal noise T=290 K ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, tx2, ch2, rx2 = run_simulation(
        cfg, rng, "2. Noise T=290 K", override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 3 – Thermal noise T=20 K (low noise receiver)
    # ==================================================================
    print("\n--- Scenario 3: Thermal noise T=20 K ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "3. Noise T=20 K", override_noise_temp_k=20.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 4 – Thermal noise T=500 K (high noise)
    # ==================================================================
    print("\n--- Scenario 4: Thermal noise T=500 K ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "4. Noise T=500 K", override_noise_temp_k=500.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 5 – Doppler 3 Hz WITHOUT correction
    # ==================================================================
    print("\n--- Scenario 5: Doppler 3 Hz (no correction) ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.apply_doppler_correction = False
    rng = np.random.default_rng(cfg.random_seed)
    r, tx5, ch5, rx5 = run_simulation(
        cfg, rng, "5. Doppler (no corr.)",
        override_doppler_hz=3.0, override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 6 – Doppler 3 Hz WITH correction
    # ==================================================================
    print("\n--- Scenario 6: Doppler 3 Hz (with correction) ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.apply_doppler_correction = True
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "6. Doppler (corrected)",
        override_doppler_hz=3.0, override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 7 – HPA IBO = 30 dB (near linear)
    # ==================================================================
    print("\n--- Scenario 7: HPA IBO = 30 dB ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "7. HPA IBO=30 dB",
        override_noise_temp_k=290.0, custom_backoff_db=30.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 8 – HPA IBO = 7 dB (moderate compression)
    # ==================================================================
    print("\n--- Scenario 8: HPA IBO = 7 dB ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, tx8, ch8, rx8 = run_simulation(
        cfg, rng, "8. HPA IBO=7 dB",
        override_noise_temp_k=290.0, custom_backoff_db=7.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 9 – HPA IBO = 1 dB (heavy saturation)
    # ==================================================================
    print("\n--- Scenario 9: HPA IBO = 1 dB ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.doppler_hz = 0.0
    rng = np.random.default_rng(cfg.random_seed)
    r, tx9, ch9, rx9 = run_simulation(
        cfg, rng, "9. HPA IBO=1 dB",
        override_noise_temp_k=290.0, custom_backoff_db=1.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 10 – I/Q imbalance WITHOUT correction
    # ==================================================================
    print("\n--- Scenario 10: I/Q imbalance (no correction) ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    cfg.apply_iq_imbalance = True
    cfg.apply_iq_correction = False
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "10. I/Q imbal. (no corr.)",
        override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 11 – I/Q imbalance WITH correction
    # ==================================================================
    print("\n--- Scenario 11: I/Q imbalance (with correction) ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    cfg.apply_iq_imbalance = True
    cfg.apply_iq_correction = True
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "11. I/Q imbal. (corrected)",
        override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 12 – Phase noise
    # ==================================================================
    print("\n--- Scenario 12: Phase noise ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    cfg.apply_phase_noise = True
    cfg.phase_noise_power_rad2 = 1e-3
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "12. Phase noise",
        override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # SCENARIO 13 – DC offset with correction
    # ==================================================================
    print("\n--- Scenario 13: DC offset (with correction) ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.hpa_input_backoff_db = 30.0
    cfg.doppler_hz = 0.0
    cfg.apply_dc_offset = True
    cfg.apply_dc_correction = True
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(
        cfg, rng, "13. DC offset (corrected)",
        override_noise_temp_k=290.0)
    scenarios.append(r)

    # ==================================================================
    # Print summary table
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"{'Scenario':<35} {'BER':>10} {'Eb/N0 (dB)':>12} {'EVM (%)':>9}")
    print("-" * 65)
    for s in scenarios:
        print(f"  {s.name:<33} {s.ber:>10.3e} {s.ebn0_db:>12.1f} {s.evm_pct:>9.1f}")
    print("=" * 65)

    # ==================================================================
    # Plots
    # ==================================================================
    print("\nGenerating plots …")

    # -- HPA characteristics --
    P.plot_hpa_characteristics(
        base_cfg.hpa_saleh_a_a, base_cfg.hpa_saleh_b_a,
        base_cfg.hpa_saleh_a_p, base_cfg.hpa_saleh_b_p,
        ibo_levels=[30, 7, 1],
        save_path=fig_path("hpa_characteristics.png"),
    )

    # -- SRRC filter --
    h = srrc_coeffs(base_cfg.rolloff, base_cfg.span, base_cfg.samples_per_symbol)
    P.plot_srrc_response(h, base_cfg.samples_per_symbol,
                         save_path=fig_path("srrc_response.png"))

    # -- Constellations: IBO=7 scenario (moderate HPA) --
    # Downsample the filtered signal to symbol rate for constellation
    delay = filter_delay(base_cfg.span, base_cfg.samples_per_symbol)

    from filters import rx_filter
    syms_before_hpa = rx_filter(tx8.filtered, h,
                                 base_cfg.samples_per_symbol, delay)[:base_cfg.num_symbols]
    syms_after_hpa  = rx_filter(tx8.after_hpa, h,
                                 base_cfg.samples_per_symbol, delay)[:base_cfg.num_symbols]
    syms_rx8        = rx8.symbols[:base_cfg.num_symbols]

    P.plot_constellations(
        syms_before_hpa, syms_after_hpa, syms_rx8,
        title_suffix="(IBO = 7 dB, T = 290 K)",
        save_path=fig_path("constellations_ibo7.png"),
    )

    # -- Constellations: IBO=1 (heavy clipping) --
    syms_before_hpa9 = rx_filter(tx9.filtered, h,
                                  base_cfg.samples_per_symbol, delay)[:base_cfg.num_symbols]
    syms_after_hpa9  = rx_filter(tx9.after_hpa, h,
                                  base_cfg.samples_per_symbol, delay)[:base_cfg.num_symbols]
    syms_rx9         = rx9.symbols[:base_cfg.num_symbols]

    P.plot_constellations(
        syms_before_hpa9, syms_after_hpa9, syms_rx9,
        title_suffix="(IBO = 1 dB – heavy saturation)",
        save_path=fig_path("constellations_ibo1.png"),
    )

    # -- Spectra --
    P.plot_spectra(
        tx2.after_hpa, ch2.signal,
        sample_rate=base_cfg.sample_rate_hz,
        title_suffix="(T = 290 K, IBO = 30 dB)",
        save_path=fig_path("spectra_nominal.png"),
    )

    # -- BER comparison bar chart --
    P.plot_ber_comparison(
        [s.name for s in scenarios],
        [s.ber for s in scenarios],
        title="BER Comparison – All Scenarios",
        save_path=fig_path("ber_comparison.png"),
    )

    # -- BER vs Eb/N0 sweep --
    print("\nRunning Eb/N0 sweep for BER curve …")
    ebn0_sweep = list(range(5, 23, 2))   # 5, 7, 9, …, 21 dB
    ebn0_list, ber_list = run_ebn0_sweep(base_cfg, ebn0_sweep)
    P.plot_ber_vs_ebn0(
        ebn0_list, ber_list,
        title="Simulated vs Theoretical BER – 16-QAM AWGN",
        save_path=fig_path("ber_vs_ebn0.png"),
    )

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("Simulation complete.\n")

    return scenarios


if __name__ == "__main__":
    main()
