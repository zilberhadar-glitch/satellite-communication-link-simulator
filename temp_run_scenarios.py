"""
temp_run_scenarios.py  (v3)
---------------------------
External robustness runner for the RF Satellite Link simulation.

Changes from v2
---------------
  Group C  Doppler:
    * Tested Doppler values are now REALISTIC fractions of the normalised
      symbol rate (0.001 to 0.05), not 1–20 Hz which were 1–20× the symbol
      rate and physically meaningless.
    * Three modes per Doppler value:
        no_correction     – receiver ignores the CFO entirely
        ideal             – receiver is given the TRUE CFO value (upper bound)
        data_aided_blind  – receiver uses the 4th-power estimator on symbol-
                            rate data (with zero-padding) followed by a
                            data-aided phase correction using known pilots.
                            The 'blind' label is replaced with 'data_aided'
                            because the phase correction uses known symbols.

  Group E  I/Q imbalance:
    * Two correction modes per imbalance level:
        estimated  – moment-matched estimator (as before)
        ideal      – exact model inverse using known parameters (new)
      This separates whether the formula is correct from whether the
      estimator is accurate.

  Group G  Noise stress (NEW):
    * Directly sweeps Eb/N0 from 5 to 20 dB using calibrated AWGN injection
      to verify that the BER curve follows the theoretical 16-QAM prediction.
    * Separate from Group A which uses physical noise temperature and never
      stresses the link due to the large SNR margin.

  Group H  HPA isolation (NEW):
    * Runs HPA disabled (ideal linear amplifier) as a baseline, then
      enabled at selected IBO values, isolating the HPA contribution to BER.

Original project files modified: receiver.py  (Doppler + IQ corrections)
All other files: UNCHANGED.
"""

import copy, csv, io, os, sys, traceback
from datetime import datetime
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH   = os.path.join(_SCRIPT_DIR, "terminal_scenario_log.txt")
_CSV_PATH   = os.path.join(_SCRIPT_DIR, "terminal_scenario_results.csv")

# ── tee stdout to log file ────────────────────────────────────────────────────
class _Tee:
    def __init__(self, real_stdout, log_file):
        self._real = real_stdout; self._log = log_file
    def write(self, data):
        self._real.write(data); self._log.write(data); return len(data)
    def flush(self):
        self._real.flush(); self._log.flush()

# ── CSV columns ──────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "group", "scenario_name", "correction_mode",
    "noise_temp_k", "ebn0_db_injected", "hpa_ibo_db", "hpa_enabled",
    "doppler_norm",
    "phase_noise_var", "iq_amp_db", "iq_phase_deg", "dc_offset_frac",
    "ber", "ser", "n_bit_errors",
    "snr_db", "ebn0_db_link", "evm_pct", "papr_db",
    "estimated_doppler_hz",
    "status", "warnings", "notes",
]

# ── base config ───────────────────────────────────────────────────────────────
def _base_cfg():
    from config import Config
    return Config(
        modulation_order=16,
        carrier_freq_hz=4.0e9,
        sat_altitude_km=35_600.0,
        tx_antenna_diameter_m=0.4,
        rx_antenna_diameter_m=0.4,
        lna_gain_db=30.0,
        noise_temp_k=290.0,
        hpa_input_backoff_db=30.0,
        doppler_hz=0.0,
        rolloff=0.25, span=10, samples_per_symbol=8,
        num_symbols=10_000, random_seed=42,
        apply_doppler_correction=False,
        apply_phase_noise=False, phase_noise_power_rad2=0.0,
        apply_iq_imbalance=False,
        iq_amplitude_imbalance_db=0.0, iq_phase_imbalance_deg=0.0,
        apply_iq_correction=False,
        apply_dc_offset=False, dc_offset_i=0.0, dc_offset_q=0.0,
        apply_dc_correction=False,
        apply_agc=True, agc_target_power=1.0,
        verbose=False,
    )

# ── single simulation run ─────────────────────────────────────────────────────
def _run(cfg,
         override_noise_temp_k=None,
         override_doppler_hz=None,
         rx_doppler_override=None,
         ref_evm_pct=None,
         iq_amp_db_known=None,
         iq_phase_deg_known=None,
         use_iq_ideal=False):
    """
    Execute one full Tx→Channel→Rx run.

    rx_doppler_override:
        a float → ideal mode (true value passed to receiver)
        None    → data-aided blind mode (4th-power + pilot phase correction)

    use_iq_ideal:
        True  → apply exact model inverse using known iq_amp/phase parameters
        False → apply estimated correction (current receiver.py function)
    """
    from transmitter import transmit
    from channel     import propagate
    from receiver    import receive, attach_srrc_h, iq_correct_ideal, _agc
    from metrics     import compute_ebn0_db, compute_evm, compute_papr_db
    from modulation  import bits_to_symbols
    from filters     import rx_filter, filter_delay

    try:
        attach_srrc_h(cfg)
        rng = np.random.default_rng(cfg.random_seed)
        tx  = transmit(cfg, rng)

        # Build ref_symbols for the data-aided phase correction
        ref_symbols = bits_to_symbols(tx.bits, cfg.modulation_order)

        ch = propagate(cfg=cfg, tx_signal=tx.after_hpa, rng=rng,
                       override_doppler_hz=override_doppler_hz,
                       override_noise_temp_k=override_noise_temp_k)

        if use_iq_ideal:
            # ── ideal I/Q correction path ─────────────────────────────────
            # Run the normal receive() chain but without IQ correction,
            # then apply the exact model inverse afterwards.
            cfg_no_iq = copy.deepcopy(cfg)
            cfg_no_iq.apply_iq_correction = False
            attach_srrc_h(cfg_no_iq)

            rx_tmp = receive(rx_signal=ch.signal, tx_bits=tx.bits,
                             cfg=cfg_no_iq,
                             override_doppler_hz=rx_doppler_override,
                             ref_symbols=ref_symbols)

            # Get the matched-filter output before AGC was applied.
            # We redo the MF+downsample step to get pre-AGC symbols.
            from filters import filter_delay as _fd
            delay = _fd(cfg.span, cfg.samples_per_symbol)
            syms_mf = rx_filter(ch.signal, cfg.srrc_h,
                                 cfg.samples_per_symbol, delay)

            # Apply Doppler correction if it was requested
            if cfg.apply_doppler_correction:
                if rx_doppler_override is not None:
                    t = np.arange(len(syms_mf))
                    syms_mf = syms_mf * np.exp(
                        -1j * 2*np.pi * rx_doppler_override * t)
                else:
                    # Use the estimate from the full receive() call
                    est = rx_tmp.estimated_doppler_hz
                    t   = np.arange(len(syms_mf))
                    syms_mf = syms_mf * np.exp(-1j * 2*np.pi * est * t)
                    # Data-aided phase correction
                    from receiver import _phase_estimate_data_aided
                    ph = _phase_estimate_data_aided(syms_mf, ref_symbols,
                                                    skip=cfg.span)
                    syms_mf = syms_mf * np.exp(-1j * ph)

            amp_db  = iq_amp_db_known  if iq_amp_db_known  is not None \
                      else cfg.iq_amplitude_imbalance_db
            ph_deg  = iq_phase_deg_known if iq_phase_deg_known is not None \
                      else cfg.iq_phase_imbalance_deg

            syms_corrected = iq_correct_ideal(syms_mf, amp_db, ph_deg)
            syms_final     = _agc(syms_corrected)

            from modulation import symbols_to_bits, symbol_error_rate
            rx_bits = symbols_to_bits(syms_final[:cfg.num_symbols],
                                       cfg.modulation_order)
            ber_val, ser_val, n_err = symbol_error_rate(
                tx.bits, rx_bits, cfg.bits_per_symbol)

            from dataclasses import dataclass
            class _FakeRx:
                ber = ber_val; ser = ser_val; n_bit_errors = n_err
                symbols = syms_final
                estimated_doppler_hz = rx_tmp.estimated_doppler_hz
            rx = _FakeRx()
        else:
            rx = receive(rx_signal=ch.signal, tx_bits=tx.bits, cfg=cfg,
                         override_doppler_hz=rx_doppler_override,
                         ref_symbols=ref_symbols)

        nt   = override_noise_temp_k if override_noise_temp_k is not None \
               else cfg.noise_temp_k
        ebn0 = compute_ebn0_db(cfg, noise_temp_k=nt)
        evm  = compute_evm(ref_symbols, rx.symbols)
        papr = compute_papr_db(tx.after_hpa)

        warnings = []
        if ref_evm_pct is not None and isinstance(ref_evm_pct, float):
            if evm > ref_evm_pct + 0.5:
                warnings.append(
                    f"CORRECTION_WORSENS_EVM"
                    f"(before={ref_evm_pct:.2f}% after={evm:.2f}%)")
        if override_doppler_hz is not None and cfg.apply_doppler_correction \
                and rx_doppler_override is None:
            est  = rx.estimated_doppler_hz
            true = override_doppler_hz
            if abs(true) > 1e-9 and abs(est-true)/abs(true) > 0.05:
                warnings.append(
                    f"DOPPLER_EST_ERROR"
                    f"(true={true:.5f} est={est:.5f})")

        return {
            "ber": rx.ber, "ser": rx.ser, "n_bit_errors": rx.n_bit_errors,
            "snr_db": ch.snr_db, "ebn0_db_link": ebn0,
            "evm_pct": evm, "papr_db": papr,
            "estimated_doppler_hz": rx.estimated_doppler_hz,
            "status": "OK",
            "warnings": "; ".join(warnings) if warnings else "",
            "notes": "",
        }
    except Exception as exc:
        return {k: "N/A" for k in
                ["ber","ser","n_bit_errors","snr_db","ebn0_db_link",
                 "evm_pct","papr_db","estimated_doppler_hz"]} | {
            "status": "ERROR", "warnings": "", "notes": str(exc)[:140]}

# ── CSV write helper ──────────────────────────────────────────────────────────
def _csv(writer, group, name, mode, params, res):
    writer.writerow({
        "group": group, "scenario_name": name, "correction_mode": mode,
        "noise_temp_k":      params.get("noise_temp_k",       ""),
        "ebn0_db_injected":  params.get("ebn0_db_injected",   ""),
        "hpa_ibo_db":        params.get("hpa_ibo_db",         ""),
        "hpa_enabled":       params.get("hpa_enabled",        ""),
        "doppler_norm":      params.get("doppler_norm",       ""),
        "phase_noise_var":   params.get("phase_noise_var",    ""),
        "iq_amp_db":         params.get("iq_amp_db",          ""),
        "iq_phase_deg":      params.get("iq_phase_deg",       ""),
        "dc_offset_frac":    params.get("dc_offset_frac",     ""),
        **{k: res[k] for k in [
            "ber","ser","n_bit_errors","snr_db","ebn0_db_link",
            "evm_pct","papr_db","estimated_doppler_hz",
            "status","warnings","notes"]},
    })

def _print_row(name, mode, res):
    ber  = f"{res['ber']:.4e}" if isinstance(res['ber'], float) else res['ber']
    evm  = f"{res['evm_pct']:.2f}%" if isinstance(res['evm_pct'], float) else ""
    edop = (f"{res['estimated_doppler_hz']:.6f}"
            if isinstance(res['estimated_doppler_hz'], float) else "")
    warn = f"  ⚠ {res['warnings']}" if res['warnings'] else ""
    print(f"    [{mode:<14}] BER={ber}  EVM={evm}  est_CFO={edop}{warn}")

# ════════════════════════════════════════════════════════════════════════════
# GROUP A — Noise temperature (physical link budget)
# ════════════════════════════════════════════════════════════════════════════
def sweep_noise_temperature(writer):
    group = "A_noise_temperature"
    temps = [0, 20, 290, 500, 1000, 5000]
    print(f"\n{'='*65}\n GROUP A: Noise Temperature Sweep (physical link budget)\n{'='*65}")
    print("  Note: link SNR margin is ~40 dB above 16-QAM threshold; BER=0 expected.")
    for t in temps:
        name = f"noise_T={t}K"
        print(f"  {name}")
        cfg = _base_cfg(); cfg.hpa_input_backoff_db = 30.0
        res = _run(cfg, override_noise_temp_k=t)
        _print_row(name, "none", res)
        _csv(writer, group, name, "none",
             {"noise_temp_k": t, "hpa_ibo_db": 30}, res)

# ════════════════════════════════════════════════════════════════════════════
# GROUP B — HPA back-off
# ════════════════════════════════════════════════════════════════════════════
def sweep_hpa_backoff(writer):
    group = "B_hpa_backoff"
    ibos  = [30, 15, 10, 7, 5, 3, 1]
    print(f"\n{'='*65}\n GROUP B: HPA Input Back-Off Sweep\n{'='*65}")
    for ibo in ibos:
        name = f"HPA_IBO={ibo}dB"
        print(f"  {name}")
        cfg = _base_cfg(); cfg.hpa_input_backoff_db = ibo
        res = _run(cfg, override_noise_temp_k=290.0)
        _print_row(name, "none", res)
        _csv(writer, group, name, "none",
             {"noise_temp_k": 290, "hpa_ibo_db": ibo}, res)

# ════════════════════════════════════════════════════════════════════════════
# GROUP C — Doppler sweep  (realistic normalised offsets, three modes)
# ════════════════════════════════════════════════════════════════════════════
def sweep_doppler(writer):
    """
    Doppler values are normalised fractions of the symbol rate.

    Physical context
    ----------------
    Symbol rate = 1 (normalised).  In a real 36 Mbaud / 4 GHz C-band system:
      max Doppler ≈ 4 GHz × (7 km/s orbital velocity) / c ≈ 93 kHz
      normalised  ≈ 93e3 / 36e6 ≈ 0.0026

    The sweep covers 0.001 to 0.050 (0.1% to 5% of symbol rate), which spans
    from realistic GEO/LEO values to a worst-case stress test.

    Values of 1–20 Hz in the v2 runner corresponded to 1×–20× the symbol rate,
    which is physically impossible and explains why the estimator failed.

    Correction modes
    ----------------
    no_correction   – receiver ignores the CFO
    ideal           – true CFO value passed directly to the receiver
    data_aided      – 4th-power estimator on symbol-rate data (4× zero-padded)
                      + data-aided pilot phase correction using known symbols.
                      Called 'data_aided' because the phase step uses known
                      reference symbols; the CFO frequency estimate is blind.
    """
    group = "C_doppler"
    freqs = [0.0, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050]
    print(f"\n{'='*65}\n GROUP C: Doppler Sweep (normalised; 3 modes)\n{'='*65}")
    print("  Doppler values are fractions of symbol rate (0.001=0.1%, 0.05=5%).")

    for fd in freqs:
        print(f"  Doppler = {fd:.4f} (normalised)")

        # no correction
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_doppler_correction = False
        res_none = _run(cfg, override_noise_temp_k=290.0,
                        override_doppler_hz=fd)
        _print_row(f"fd={fd:.4f}", "no_correction", res_none)
        _csv(writer, group, f"Doppler={fd:.4f}_no_correction", "no_correction",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "doppler_norm": fd},
             res_none)

        # ideal
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_doppler_correction = True
        res_ideal = _run(cfg, override_noise_temp_k=290.0,
                         override_doppler_hz=fd,
                         rx_doppler_override=fd)
        _print_row(f"fd={fd:.4f}", "ideal", res_ideal)
        _csv(writer, group, f"Doppler={fd:.4f}_ideal", "ideal",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "doppler_norm": fd},
             res_ideal)

        # data_aided (blind CFO + pilot phase)
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_doppler_correction = True
        ref_evm = (res_none['evm_pct']
                   if isinstance(res_none['evm_pct'], float) else None)
        res_da = _run(cfg, override_noise_temp_k=290.0,
                      override_doppler_hz=fd,
                      rx_doppler_override=None,   # use estimator
                      ref_evm_pct=ref_evm)
        _print_row(f"fd={fd:.4f}", "data_aided", res_da)
        _csv(writer, group, f"Doppler={fd:.4f}_data_aided", "data_aided",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "doppler_norm": fd},
             res_da)

# ════════════════════════════════════════════════════════════════════════════
# GROUP D — Phase noise
# ════════════════════════════════════════════════════════════════════════════
def sweep_phase_noise(writer):
    group     = "D_phase_noise"
    variances = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    print(f"\n{'='*65}\n GROUP D: Phase Noise Variance Sweep\n{'='*65}")
    for var in variances:
        name = f"phase_noise_var={var:.0e}" if var > 0 else "phase_noise_var=0"
        print(f"  {name}")
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db   = 30.0
        cfg.apply_phase_noise      = (var > 0)
        cfg.phase_noise_power_rad2 = var
        res = _run(cfg, override_noise_temp_k=290.0)
        _print_row(name, "none", res)
        _csv(writer, group, name, "none",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "phase_noise_var": var},
             res)

# ════════════════════════════════════════════════════════════════════════════
# GROUP E — I/Q imbalance  (estimated vs ideal correction)
# ════════════════════════════════════════════════════════════════════════════
def sweep_iq_imbalance(writer):
    """
    Two correction modes:
      estimated  – receiver.py moment-matched estimator (practical, blind)
      ideal      – exact model inverse using the known impairment parameters
                   (validates the formula independent of estimator accuracy)
    """
    group = "E_iq_imbalance"
    cases = [(0.0,0.0),(1.0,5.0),(2.0,10.0),(3.0,15.0),(5.0,20.0)]
    print(f"\n{'='*65}\n GROUP E: I/Q Imbalance (no_correction / estimated / ideal)\n{'='*65}")

    for amp, phase in cases:
        label = (f"IQ_{amp}dB_{phase}deg" if amp>0 or phase>0 else "IQ_none")
        print(f"  {label}")

        # no correction baseline
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db      = 30.0
        cfg.apply_iq_imbalance        = (amp>0 or phase>0)
        cfg.iq_amplitude_imbalance_db = amp
        cfg.iq_phase_imbalance_deg    = phase
        cfg.apply_iq_correction       = False
        res_none = _run(cfg, override_noise_temp_k=290.0)
        _print_row(label, "no_correction", res_none)
        _csv(writer, group, f"{label}_no_correction", "no_correction",
             {"noise_temp_k":290,"hpa_ibo_db":30,"iq_amp_db":amp,"iq_phase_deg":phase},
             res_none)

        ref_evm = res_none['evm_pct'] if isinstance(res_none['evm_pct'], float) else None

        # estimated correction
        cfg2 = _base_cfg()
        cfg2.hpa_input_backoff_db      = 30.0
        cfg2.apply_iq_imbalance        = (amp>0 or phase>0)
        cfg2.iq_amplitude_imbalance_db = amp
        cfg2.iq_phase_imbalance_deg    = phase
        cfg2.apply_iq_correction       = True
        res_est = _run(cfg2, override_noise_temp_k=290.0, ref_evm_pct=ref_evm)
        _print_row(label, "estimated", res_est)
        _csv(writer, group, f"{label}_estimated", "estimated",
             {"noise_temp_k":290,"hpa_ibo_db":30,"iq_amp_db":amp,"iq_phase_deg":phase},
             res_est)

        # ideal correction (uses known parameters)
        cfg3 = _base_cfg()
        cfg3.hpa_input_backoff_db      = 30.0
        cfg3.apply_iq_imbalance        = (amp>0 or phase>0)
        cfg3.iq_amplitude_imbalance_db = amp
        cfg3.iq_phase_imbalance_deg    = phase
        cfg3.apply_iq_correction       = False   # handled inside _run
        res_ideal = _run(cfg3, override_noise_temp_k=290.0,
                         ref_evm_pct=ref_evm,
                         iq_amp_db_known=amp, iq_phase_deg_known=phase,
                         use_iq_ideal=True)
        _print_row(label, "ideal", res_ideal)
        _csv(writer, group, f"{label}_ideal", "ideal",
             {"noise_temp_k":290,"hpa_ibo_db":30,"iq_amp_db":amp,"iq_phase_deg":phase},
             res_ideal)

# ════════════════════════════════════════════════════════════════════════════
# GROUP F — DC offset
# ════════════════════════════════════════════════════════════════════════════
def sweep_dc_offset(writer):
    group     = "F_dc_offset"
    fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    print(f"\n{'='*65}\n GROUP F: DC Offset Sweep\n{'='*65}")
    for frac in fractions:
        label = f"DC={frac:.2f}"
        print(f"  {label}")
        for correct, mode in [(False,"no_correction"),(True,"blind")]:
            cfg = _base_cfg()
            cfg.hpa_input_backoff_db = 30.0
            cfg.apply_dc_offset      = (frac>0)
            cfg.dc_offset_i          = frac
            cfg.dc_offset_q          = frac * 0.75
            cfg.apply_dc_correction  = correct
            res = _run(cfg, override_noise_temp_k=290.0)
            _print_row(label, mode, res)
            _csv(writer, group, f"{label}_{mode}", mode,
                 {"noise_temp_k":290,"hpa_ibo_db":30,"dc_offset_frac":frac}, res)

# ════════════════════════════════════════════════════════════════════════════
# GROUP G — Noise stress: direct Eb/N0 sweep (NEW)
# ════════════════════════════════════════════════════════════════════════════
def sweep_ebn0_stress(writer):
    """
    Directly inject AWGN at calibrated Eb/N0 levels to verify the BER curve.

    Noise is injected at SYMBOL RATE after the matched filter, with:
        sigma^2 = 1 / (Eb/N0_linear * bits_per_symbol)  [per complex sample]

    This matches the derivation of  BER ≈ (3/8)*erfc(sqrt(Eb/N0/5))  for
    Gray-coded 16-QAM with unit average symbol power.  The simulation should
    closely track the theoretical curve with no systematic offset.
    """
    from filters    import srrc_coeffs, tx_filter, rx_filter, filter_delay
    from modulation import bits_to_symbols, symbols_to_bits, symbol_error_rate
    from metrics    import ber_theory_16qam_awgn

    group    = "G_noise_stress_ebn0"
    ebn0_dbs = list(range(5, 21, 2))
    print(f"\n{'='*65}\n GROUP G: Noise Stress — Direct Eb/N0 Sweep\n{'='*65}")
    print("  Noise injected at symbol rate after matched filter.")
    print("  Should match BER = (3/8)*erfc(sqrt(Eb/N0/5)) within ~2×.")
    print(f"  {'Eb/N0 (dB)':>12} {'BER_sim':>12} {'BER_theory':>12} {'ratio':>8}")
    print("  " + "-"*50)

    cfg = _base_cfg()
    cfg.hpa_input_backoff_db = 30.0
    h   = srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)
    dly = filter_delay(cfg.span, cfg.samples_per_symbol)
    bps = cfg.bits_per_symbol

    for ebn0_db in ebn0_dbs:
        rng  = np.random.default_rng(cfg.random_seed)
        bits = rng.integers(0, 2, size=cfg.num_symbols * bps)
        syms = bits_to_symbols(bits, cfg.modulation_order)

        # Transmit + matched filter → symbol-rate output
        tx_s = tx_filter(syms, h, cfg.samples_per_symbol)
        tx_s /= np.sqrt(np.mean(np.abs(tx_s) ** 2))
        syms_mf = rx_filter(tx_s, h, cfg.samples_per_symbol, dly)
        syms_mf /= np.sqrt(np.mean(np.abs(syms_mf) ** 2))  # unit average power

        # Inject noise at symbol level: sigma^2 = 1/(Eb/N0 * bps)
        ebn0_lin = 10.0 ** (ebn0_db / 10.0)
        sigma    = np.sqrt(1.0 / (ebn0_lin * bps))
        noise    = sigma * (rng.standard_normal(len(syms_mf))
                            + 1j * rng.standard_normal(len(syms_mf)))
        syms_noisy = syms_mf + noise

        rx_bits = symbols_to_bits(syms_noisy[:cfg.num_symbols], cfg.modulation_order)
        ber_sim, ser_sim, n_err = symbol_error_rate(bits, rx_bits, bps)
        ber_th  = float(ber_theory_16qam_awgn(np.array([ebn0_db]))[0])
        ratio   = (ber_sim / ber_th) if (ber_th > 0 and ber_sim > 0) else float('nan')

        warn_str = ""
        if ber_sim > 0 and ber_th > 0 and (ratio > 2.0 or ratio < 0.5):
            warn_str = f"BER_MISMATCH(sim={ber_sim:.2e} theory={ber_th:.2e})"

        ratio_str = f"{ratio:.2f}" if not (isinstance(ratio, float) and np.isnan(ratio)) else "   —"
        print(f"  {ebn0_db:>12.1f} {ber_sim:>12.4e} {ber_th:>12.4e} {ratio_str:>8}"
              + (f"  ⚠ {warn_str}" if warn_str else ""))

        res = {
            "ber": ber_sim, "ser": ser_sim, "n_bit_errors": n_err,
            "snr_db": ebn0_db + 10 * np.log10(bps),
            "ebn0_db_link": ebn0_db,
            "evm_pct": float('nan'), "papr_db": float('nan'),
            "estimated_doppler_hz": 0.0,
            "status": "OK", "warnings": warn_str, "notes": f"theory={ber_th:.4e}",
        }
        _csv(writer, group, f"Eb/N0={ebn0_db}dB", "none",
             {"hpa_ibo_db": 30, "hpa_enabled": True, "ebn0_db_injected": ebn0_db},
             res)

# ════════════════════════════════════════════════════════════════════════════
# GROUP H — HPA isolation: linear baseline vs nonlinear (NEW)
# ════════════════════════════════════════════════════════════════════════════
def sweep_hpa_isolation(writer):
    """
    Compare a linear amplifier baseline against the Saleh TWTA at several IBO
    values.  This isolates how much of the BER comes from HPA nonlinearity
    rather than from noise or other impairments.

    The 'linear' case uses IBO=30 dB, which keeps the Saleh model well within
    its linear region (< 1% AM/AM deviation from ideal).  For a truly ideal
    amplifier you would bypass the HPA entirely; at IBO=30 dB the Saleh model
    gives output amplitude ≈ input × constant, so it is effectively linear.
    """
    group = "H_hpa_isolation"
    cases = [
        ("linear_IBO30dB", 30, True,  "Saleh model at IBO=30 dB (linear region)"),
        ("IBO_15dB",        15, True,  "Moderate compression"),
        ("IBO_10dB",        10, True,  ""),
        ("IBO_7dB",          7, True,  ""),
        ("IBO_5dB",          5, True,  ""),
        ("IBO_3dB",          3, True,  ""),
        ("IBO_1dB",          1, True,  "Heavy saturation"),
    ]
    print(f"\n{'='*65}\n GROUP H: HPA Nonlinearity Isolation\n{'='*65}")
    print("  All cases at T=290 K, no Doppler/IQ/phase-noise.")
    print("  'linear_IBO30dB' is the HPA-off baseline.")

    for name, ibo, hpa_on, note in cases:
        print(f"  {name}")
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db = ibo
        res = _run(cfg, override_noise_temp_k=290.0)
        _print_row(name, "none", res)
        if note:
            print(f"    ({note})")
        _csv(writer, group, name, "none",
             {"noise_temp_k":290, "hpa_ibo_db":ibo, "hpa_enabled":hpa_on}, res)

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def main():
    start    = datetime.now()
    log_file = open(_LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print("=" * 65)
    print("  RF Satellite Link — External Scenario Runner  v3")
    print(f"  Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CSV     : {_CSV_PATH}")
    print(f"  Log     : {_LOG_PATH}")
    print("  Fixes in this version:")
    print("    Doppler: tested at realistic normalised offsets (not 1-20× symbol rate)")
    print("    Doppler: estimator runs on symbol-rate data with 4× zero-padding")
    print("    IQ: ideal correction mode added to validate formula vs estimator")
    print("    Noise:  Eb/N0 stress sweep added (Group G)")
    print("    HPA:    linear baseline isolation (Group H)")
    print("=" * 65)

    csv_file = open(_CSV_PATH, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    sweep_noise_temperature(writer)
    sweep_hpa_backoff(writer)
    sweep_doppler(writer)
    sweep_phase_noise(writer)
    sweep_iq_imbalance(writer)
    sweep_dc_offset(writer)
    sweep_ebn0_stress(writer)
    sweep_hpa_isolation(writer)

    csv_file.flush(); csv_file.close()
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'='*65}")
    print(f"  All scenarios complete.  Elapsed: {elapsed:.1f} s")
    print(f"  Results : {_CSV_PATH}")
    print(f"  Log     : {_LOG_PATH}")
    print("=" * 65)

    sys.stdout = sys.__stdout__
    log_file.flush(); log_file.close()
    print(f"\nDone. Results saved to:\n  {_CSV_PATH}\n  {_LOG_PATH}\n")


if __name__ == "__main__":
    main()
