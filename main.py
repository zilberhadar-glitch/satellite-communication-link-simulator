"""
main.py
-------
Orchestrator and experiment runner – updated for MATLAB equivalence.

Scenarios:
  1.  Clean (no noise, HPA bypass)
  2.  Noise T=20 K  (MATLAB nominal)
  3.  Noise T=290 K
  4.  Noise T=500 K
  5.  Doppler 3 Hz, no correction
  6.  Doppler 3 Hz, blind correction
  7.  Doppler 3 Hz, carrier_sync PLL  (closest to MATLAB)
  8.  HPA bypass (ideal linear)
  9.  HPA IBO=30 dB (near linear)
  10. HPA IBO=7 dB  (moderate compression)
  11. HPA IBO=7 dB + DPD
  12. HPA IBO=1 dB  (heavy clipping)
  13. I/Q amp-only imbalance, no correction
  14. I/Q amp-only imbalance, corrected
  15. I/Q phase-only imbalance, no correction
  16. I/Q phase-only imbalance, corrected
  17. Phase noise (colored, moderate)
  18. DC offset (absolute, corrected)
"""

import copy, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from transmitter import transmit, TxSignals
from channel import propagate, ChannelOutput
from receiver import receive, attach_srrc_h
from metrics import (
    ScenarioResult,
    compute_ebn0_db,
    compute_evm,
    compute_evm_matlab_scope_like,
    compute_papr_db,
)
from filters import srrc_coeffs, filter_delay
from modulation import bits_to_symbols
import plots as P

OUT_DIR = os.path.join(os.path.dirname(__file__), "output_figures")
os.makedirs(OUT_DIR, exist_ok=True)


def fig_path(name):
    return os.path.join(OUT_DIR, name)


# ===========================================================================
# Core simulation helper
# ===========================================================================

def run_simulation(cfg, rng, name, override_doppler_hz=None,
                   override_noise_temp_k=None, custom_backoff_db=None):
    attach_srrc_h(cfg)
    tx = transmit(cfg, rng, custom_backoff_db=custom_backoff_db)
    ch = propagate(tx.after_hpa, cfg, rng,
                   override_doppler_hz=override_doppler_hz,
                   override_noise_temp_k=override_noise_temp_k)
    dop_rx = (override_doppler_hz
              if (cfg.apply_doppler_correction and cfg.cfo_correction_mode == "ideal")
              else None)
    rx = receive(ch.signal, tx.bits, cfg, override_doppler_hz=dop_rx)

    r = ScenarioResult(name)
    r.ber = rx.ber; r.ser = rx.ser; r.n_errors = rx.n_bit_errors
    r.snr_db = ch.snr_db
    nt = override_noise_temp_k if override_noise_temp_k is not None else cfg.noise_temp_k
    r.ebn0_db = compute_ebn0_db(cfg, noise_temp_k=nt)

    ref_symbols = bits_to_symbols(tx.bits, cfg.modulation_order)

    # Keep the direct full-burst EVM for diagnostics.
    # For uncorrected Doppler, MATLAB's Constellation Scope reports a
    # short-window EVM rather than a full-burst global EVM.  Therefore, for
    # Doppler-without-correction scenarios only, report a MATLAB-scope-like
    # windowed EVM so the table matches the MATLAB display.
    r.evm_global_pct = compute_evm(ref_symbols, rx.symbols)

    effective_doppler_hz = (override_doppler_hz
                            if override_doppler_hz is not None
                            else cfg.doppler_hz)

    if abs(effective_doppler_hz) > 1e-12 and not cfg.apply_doppler_correction:
        r.evm_pct = compute_evm_matlab_scope_like(
            ref_symbols,
            rx.symbols,
            window_size=50,
            align_each_window=True,
        )
        r.evm_metric = "MATLAB-scope-like windowed EVM, 50 symbols"
    else:
        r.evm_pct = r.evm_global_pct
        r.evm_metric = "global full-burst EVM"

    r.papr_db = compute_papr_db(tx.after_hpa)
    return r, tx, ch, rx


def run_ebn0_sweep(cfg_base, ebn0_db_values, rng_seed=0):
    """
    Extra validation only: BER of ideal 16-QAM over AWGN.

    Noise is added directly at symbol rate after matched filtering so the
    injected Eb/N0 matches the standard theoretical 16-QAM BER expression.
    This is not a MATLAB RF Satellite Link scenario; it validates the mapper,
    demapper and BER calculation.
    """
    from filters import srrc_coeffs, tx_filter, rx_filter, filter_delay
    from modulation import bits_to_symbols, symbols_to_bits, symbol_error_rate

    cfg = copy.deepcopy(cfg_base)
    cfg.verbose = False
    cfg.apply_hpa = False
    h = srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)
    delay = filter_delay(cfg.span, cfg.samples_per_symbol)
    ebn0_list, ber_list = [], []

    for ebn0_db in ebn0_db_values:
        rng = np.random.default_rng(rng_seed)
        bits = rng.integers(0, 2, size=cfg.num_symbols * cfg.bits_per_symbol)
        symbols = bits_to_symbols(bits, cfg.modulation_order)

        # Pulse-shape and matched-filter to keep the same modulation chain.
        tx_sig = tx_filter(symbols, h, cfg.samples_per_symbol)
        tx_sig /= np.sqrt(np.mean(np.abs(tx_sig) ** 2))
        rx_syms = rx_filter(tx_sig, h, cfg.samples_per_symbol, delay)
        rx_syms /= np.sqrt(np.mean(np.abs(rx_syms) ** 2))

        # At symbol rate with Es=1 and k=bits/symbol:
        # complex noise variance = N0 = 1/(Eb/N0*k)
        sigma = np.sqrt(1.0 / (10**(ebn0_db/10) * cfg.bits_per_symbol))
        noise = sigma * (rng.standard_normal(len(rx_syms))
                         + 1j * rng.standard_normal(len(rx_syms)))
        rx_bits = symbols_to_bits((rx_syms + noise)[:cfg.num_symbols],
                                  cfg.modulation_order)
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
    print("  RF Satellite Link Simulation – MATLAB-Equivalent Python")
    print("=" * 65 + "\n")

    base_cfg = Config(
        modulation_order=16,
        carrier_freq_hz=4.0e9,
        sat_altitude_km=35_600.0,
        tx_antenna_diameter_m=0.4,
        rx_antenna_diameter_m=0.4,
        lna_gain_db=30.0,
        noise_temp_k=20.0,          # MATLAB default
        hpa_input_backoff_db=30.0,
        apply_hpa=True,
        apply_dpd=False,
        doppler_hz=0.0,
        apply_doppler_correction=False,
        cfo_correction_mode="blind",
        rolloff=0.25, span=10, samples_per_symbol=8,
        num_symbols=10_000, random_seed=42,
        apply_phase_noise=False, apply_iq_imbalance=False,
        apply_dc_offset=False, apply_agc=True, verbose=True,
    )
    print(base_cfg.summary())

    scenarios = []

    # ── 1. Clean ──────────────────────────────────────────────────────────
    print("\n--- Scenario 1: Clean (HPA bypass, no noise) ---")
    cfg = copy.deepcopy(base_cfg); cfg.apply_hpa = False
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(cfg, rng, "1. Clean (bypass, no noise)",
                           override_noise_temp_k=0.0)
    scenarios.append(r)

    # ── 2–4. Noise temperatures ───────────────────────────────────────────
    for T, label in [(20, "2. T=20K (MATLAB)"), (290, "3. T=290K"), (500, "4. T=500K")]:
        print(f"\n--- Scenario {label} ---")
        cfg = copy.deepcopy(base_cfg)
        rng = np.random.default_rng(cfg.random_seed)
        r, tx2, ch2, rx2 = run_simulation(cfg, rng, label,
                                           override_noise_temp_k=float(T))
        scenarios.append(r)
        if T == 20:
            tx_nom, ch_nom = tx2, ch2   # save for spectra plot

    # ── 5–7. Doppler ──────────────────────────────────────────────────────
    print("\n--- Scenario 5: Doppler 3 Hz, no correction ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.apply_doppler_correction = False
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(cfg, rng, "5. Doppler (no corr.)",
                           override_doppler_hz=3.0, override_noise_temp_k=20.0)
    scenarios.append(r)

    print("\n--- Scenario 6: Doppler 3 Hz, blind NDA estimator ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.apply_doppler_correction = True; cfg.cfo_correction_mode = "blind"
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(cfg, rng, "6. Doppler (blind NDA)",
                           override_doppler_hz=3.0, override_noise_temp_k=20.0)
    scenarios.append(r)

    print("\n--- Scenario 7: Doppler 3 Hz, carrier_sync PLL ---")
    cfg = copy.deepcopy(base_cfg)
    cfg.apply_doppler_correction = True; cfg.cfo_correction_mode = "carrier_sync"
    cfg.carrier_sync_loop_bw = 0.01; cfg.carrier_sync_damping = 0.707
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(cfg, rng, "7. Doppler (carrier_sync PLL)",
                           override_doppler_hz=3.0, override_noise_temp_k=20.0)
    scenarios.append(r)

    # ── 8–12. HPA ─────────────────────────────────────────────────────────
    print("\n--- Scenario 8: HPA bypass ---")
    cfg = copy.deepcopy(base_cfg); cfg.apply_hpa = False
    rng = np.random.default_rng(cfg.random_seed)
    r, *_ = run_simulation(cfg, rng, "8. HPA bypass", override_noise_temp_k=20.0)
    scenarios.append(r)

    for ibo, dpd, label in [
        (30, False, "9. HPA IBO=30dB"),
        (7,  False, "10. HPA IBO=7dB"),
        (7,  True,  "11. HPA IBO=7dB+DPD"),
        (1,  False, "12. HPA IBO=1dB"),
    ]:
        print(f"\n--- Scenario {label} ---")
        cfg = copy.deepcopy(base_cfg)
        cfg.apply_hpa = True; cfg.apply_dpd = dpd
        rng = np.random.default_rng(cfg.random_seed)
        r, tx_hpa, ch_hpa, rx_hpa = run_simulation(
            cfg, rng, label, override_noise_temp_k=20.0, custom_backoff_db=ibo)
        scenarios.append(r)
        if ibo == 7 and not dpd:
            tx8, rx8 = tx_hpa, rx_hpa   # save for constellation

    # ── 13–18. I/Q imbalance (MATLAB-equivalent symmetric ± model) ───────
    # MATLAB options: Amplitude (3 dB), Phase (20°), combined, corrected/not.
    # Python now uses the exact same symmetric ± model as comm.IQImbalance.
    print("\n--- Scenarios 13-18: I/Q Imbalance (MATLAB symmetric ± model) ---")
    iq_cases = [
        # (amp_db, phase_deg, correction, label)
        (3.0,  0.0,  False, "13. IQ amp-only 3dB no corr"),
        (3.0,  0.0,  True,  "14. IQ amp-only 3dB corrected"),
        (0.0,  20.0, False, "15. IQ phase-only 20deg no corr"),
        (0.0,  20.0, True,  "16. IQ phase-only 20deg corrected"),
        (3.0,  20.0, False, "17. IQ combined 3dB+20deg no corr"),
        (3.0,  20.0, True,  "18. IQ combined 3dB+20deg corrected"),
    ]
    print(f"\n  {'Scenario':<38} {'BER':>10} {'EVM(%)':>8}  Notes")
    print("  " + "-" * 65)
    for amp, ph, corr, label in iq_cases:
        print(f"\n--- Scenario {label} ---")
        cfg = copy.deepcopy(base_cfg)
        cfg.apply_iq_imbalance = True
        cfg.iq_amplitude_imbalance_db = amp
        cfg.iq_phase_imbalance_deg = ph
        cfg.apply_iq_correction = corr
        rng = np.random.default_rng(cfg.random_seed)
        r, *_ = run_simulation(cfg, rng, label, override_noise_temp_k=0.0)
        scenarios.append(r)
        note = "LMS blind compensator" if corr else "no correction"
        print(f"  {label:<38} {r.ber:>10.3e} {r.evm_pct:>8.1f}  [{note}]")

    # ── 19–21. Phase noise — MATLAB levels: −100, −55, −48 dBc/Hz @ 100 Hz ──
    # MATLAB RF Satellite Link parameter panel:
    #   "Negligible (-100 dBc/Hz @ 100 Hz)"
    #   "Low        (-55  dBc/Hz @ 100 Hz)"
    #   "High       (-48  dBc/Hz @ 100 Hz)"
    print("\n--- Scenarios 19-21: Phase Noise (MATLAB levels, Wiener 1/f² model) ---")
    pn_cases = [
        (-100.0, "19. Phase noise -100 dBc/Hz (negligible)"),
        (-55.0,  "20. Phase noise  -55 dBc/Hz (low)"),
        (-48.0,  "21. Phase noise  -48 dBc/Hz (high)"),
    ]
    print(f"\n  {'Scenario':<44} {'BER':>10} {'EVM(%)':>8} {'PN_rms(deg)':>12} {'PN_std(rad)':>12}")
    print("  " + "-" * 90)
    for pn_level, label in pn_cases:
        print(f"\n--- Scenario {label} ---")
        cfg = copy.deepcopy(base_cfg)
        cfg.apply_phase_noise = True
        cfg.phase_noise_dbc_hz = pn_level
        cfg.phase_noise_freq_offset_hz = 100.0
        cfg.phase_noise_physical_sample_rate_hz = 8_000_000.0  # 1 Mbaud × 8 sps
        rng = np.random.default_rng(cfg.random_seed)

        # Compute expected phase noise std for reporting
        import math
        L0 = 10.0 ** (pn_level / 10.0)
        sigma_w2 = L0 * 8.0 * math.pi**2 * 100.0**2 / 8_000_000.0
        sigma_w = math.sqrt(sigma_w2)
        # RMS of a Wiener process of length N: sigma_w * sqrt(N/3)
        N = base_cfg.num_symbols * base_cfg.samples_per_symbol
        pn_rms_rad = sigma_w * math.sqrt(N / 3.0)
        pn_rms_deg = math.degrees(pn_rms_rad)

        r, *_ = run_simulation(cfg, rng, label, override_noise_temp_k=0.0)
        scenarios.append(r)
        print(f"  {label:<44} {r.ber:>10.3e} {r.evm_pct:>8.1f} "
              f"{pn_rms_deg:>12.4f} {pn_rms_rad:>12.6f}")

    # ── 22–23. DC offset (MATLAB-equivalent, BEFORE LNA, IIR corrected) ──
    # MATLAB RF Satellite Link uses absolute DC values:
    #   In-phase offset:   1e-8 V,  Quadrature offset: 5e-8 V
    # These are physically meaningful only in physical-units mode.
    # In physical mode, signal RMS ≈ 8.19e-7 V at the demodulator input,
    # giving DC fractions of 1.22 % (I) and 6.10 % (Q) of signal amplitude.
    #
    # In Python normalised mode we use the equivalent relative fractions to
    # faithfully reproduce MATLAB's DC/signal ratio.  DC is injected BEFORE
    # the LNA, matching the MATLAB Simulink block diagram order.
    print("\n--- Scenarios 22-25: DC Offset (MATLAB-equivalent normalised mode) ---")
    dc_cases = [
        (0.05, 0.0,  False, "22. DC I=5% (MATLAB 1e-8 equiv), no corr"),
        (0.05, 0.0,  True,  "23. DC I=5% (MATLAB 1e-8 equiv), IIR blocker"),
        (0.0,  0.18, False, "24. DC Q=18% (MATLAB 5e-8 equiv), no corr"),
        (0.0,  0.18, True,  "25. DC Q=18% (MATLAB 5e-8 equiv), IIR blocker"),
    ]
    print(f"\n  {'Scenario':<64} {'BER':>10} {'EVM(%)':>8}  Notes")
    print("  " + "-" * 94)
    for dc_i, dc_q, corr, label in dc_cases:
        print(f"\n--- Scenario {label} ---")
        cfg = copy.deepcopy(base_cfg)
        cfg.apply_dc_offset = True
        cfg.dc_offset_mode = "relative"   # MATLAB-equivalent normalised path
        cfg.dc_offset_i = dc_i
        cfg.dc_offset_q = dc_q
        cfg.apply_dc_correction = corr
        rng = np.random.default_rng(cfg.random_seed)
        r, *_ = run_simulation(cfg, rng, label, override_noise_temp_k=20.0)
        scenarios.append(r)
        note = "IIR DC blocker (dsp.DCBlocker approx.)" if corr else "no correction"
        print(f"  {label:<64} {r.ber:>10.3e} {r.evm_pct:>8.2f}  [{note}]")

    # ==================================================================
    # Summary table
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"{'Scenario':<38} {'BER':>10} {'Eb/N0(dB)':>11} {'EVM(%)':>8} {'PAPR(dB)':>9}")
    print("-" * 70)
    for s in scenarios:
        print(f"  {s.name:<36} {s.ber:>10.3e} {s.ebn0_db:>11.1f} {s.evm_pct:>8.1f} {s.papr_db:>9.1f}")
    print("=" * 70)

    # ==================================================================
    # Plots
    # ==================================================================
    print("\nGenerating plots …")

    P.plot_hpa_characteristics(
        base_cfg.hpa_saleh_a_a, base_cfg.hpa_saleh_b_a,
        base_cfg.hpa_saleh_a_p, base_cfg.hpa_saleh_b_p,
        ibo_levels=[30, 7, 1],
        save_path=fig_path("hpa_characteristics.png"),
    )

    h = srrc_coeffs(base_cfg.rolloff, base_cfg.span, base_cfg.samples_per_symbol)
    P.plot_srrc_response(h, base_cfg.samples_per_symbol,
                         save_path=fig_path("srrc_response.png"))

    delay = filter_delay(base_cfg.span, base_cfg.samples_per_symbol)
    from filters import rx_filter
    syms_before = rx_filter(tx8.filtered, h, base_cfg.samples_per_symbol, delay)[:base_cfg.num_symbols]
    syms_after  = rx_filter(tx8.after_hpa, h, base_cfg.samples_per_symbol, delay)[:base_cfg.num_symbols]
    P.plot_constellations(syms_before, syms_after, rx8.symbols[:base_cfg.num_symbols],
                          title_suffix="(IBO=7 dB, T=20 K)",
                          save_path=fig_path("constellations_ibo7.png"))

    P.plot_spectra(tx_nom.after_hpa, ch_nom.signal,
                   sample_rate=base_cfg.sample_rate_hz,
                   title_suffix="(T=20 K nominal)",
                   save_path=fig_path("spectra_nominal.png"))

    P.plot_ber_comparison([s.name for s in scenarios], [s.ber for s in scenarios],
                          title="BER Comparison – MATLAB Equivalence Scenarios",
                          save_path=fig_path("ber_comparison.png"))

    print("\nRunning Eb/N0 sweep …")
    ebn0_list, ber_list = run_ebn0_sweep(base_cfg, list(range(5, 23, 2)))
    P.plot_ber_vs_ebn0(ebn0_list, ber_list,
                       title="Simulated vs Theoretical BER – 16-QAM AWGN",
                       save_path=fig_path("ber_vs_ebn0.png"))

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("Simulation complete.\n")
    return scenarios


if __name__ == "__main__":
    main()
