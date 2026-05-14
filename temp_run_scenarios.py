"""
temp_run_scenarios.py  (v2)
---------------------------
External scenario runner for the RF Satellite Link simulation.

Changes from v1
---------------
  * Doppler sweep now runs three modes per frequency:
      no_correction  – channel has Doppler, receiver ignores it
      ideal          – receiver receives the TRUE Doppler value (upper bound)
      blind          – receiver uses the 4th-power NDA estimator (realistic)
  * CSV gains a  correction_mode  column  (none / ideal / blind)
  * A  warnings  column flags cases where correction worsens EVM or BER,
    or where the blind Doppler estimate diverges from the true value.
  * I/Q sweep uses the fixed model-matched correction in receiver.py.

Original project files modified: receiver.py  (bug-fix only, see notes)
All other files (main.py, config.py, channel.py, …): UNCHANGED.
"""

import copy, csv, io, os, sys, traceback
from datetime import datetime
import numpy as np

# ── output paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH   = os.path.join(_SCRIPT_DIR, "terminal_scenario_log.txt")
_CSV_PATH   = os.path.join(_SCRIPT_DIR, "terminal_scenario_results.csv")

# ── tee stdout → log file ────────────────────────────────────────────────────
class _Tee:
    def __init__(self, real_stdout, log_file):
        self._real = real_stdout
        self._log  = log_file
    def write(self, data):
        self._real.write(data); self._log.write(data); return len(data)
    def flush(self):
        self._real.flush(); self._log.flush()

# ── CSV columns ──────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "group", "scenario_name", "correction_mode",
    "noise_temp_k", "hpa_ibo_db",
    "doppler_hz_true", "phase_noise_var",
    "iq_amp_db", "iq_phase_deg",
    "dc_offset_frac",
    "ber", "ser", "n_bit_errors",
    "snr_db", "ebn0_db", "evm_pct", "papr_db",
    "estimated_doppler_hz",
    "status", "warnings", "notes",
]

# ── build a clean base config ────────────────────────────────────────────────
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

# ── single run ────────────────────────────────────────────────────────────────
def _run(cfg,
         override_noise_temp_k=None,
         override_doppler_hz=None,
         rx_doppler_override=None,   # None → blind estimator
         ref_evm_pct=None):          # EVM of the un-corrected baseline
    """
    Execute one Tx→Channel→Rx simulation.
    rx_doppler_override:
        a float  → passed as the known Doppler to the receiver (ideal mode)
        None     → receiver uses its blind 4th-power estimator
    """
    from transmitter import transmit
    from channel     import propagate
    from receiver    import receive, attach_srrc_h
    from metrics     import compute_ebn0_db, compute_evm, compute_papr_db
    from modulation  import bits_to_symbols
    try:
        attach_srrc_h(cfg)
        rng = np.random.default_rng(cfg.random_seed)

        tx = transmit(cfg, rng)
        ch = propagate(cfg=cfg, tx_signal=tx.after_hpa, rng=rng,
                       override_doppler_hz=override_doppler_hz,
                       override_noise_temp_k=override_noise_temp_k)
        rx = receive(rx_signal=ch.signal, tx_bits=tx.bits, cfg=cfg,
                     override_doppler_hz=rx_doppler_override)

        nt    = override_noise_temp_k if override_noise_temp_k is not None \
                else cfg.noise_temp_k
        ebn0  = compute_ebn0_db(cfg, noise_temp_k=nt)
        ref   = bits_to_symbols(tx.bits, cfg.modulation_order)
        evm   = compute_evm(ref, rx.symbols)
        papr  = compute_papr_db(tx.after_hpa)

        # ── build warning string ─────────────────────────────────────
        warnings = []

        # correction mode worsens EVM?
        if ref_evm_pct is not None and evm > ref_evm_pct + 0.5:
            warnings.append(
                f"CORRECTION_WORSENS_EVM "
                f"(before={ref_evm_pct:.2f}% after={evm:.2f}%)")

        # Doppler estimate diverges?
        if override_doppler_hz is not None and cfg.apply_doppler_correction:
            est  = rx.estimated_doppler_hz
            true = override_doppler_hz
            if abs(true) > 0 and abs(est - true) / (abs(true) + 1e-9) > 0.10:
                warnings.append(
                    f"DOPPLER_EST_ERROR "
                    f"(true={true:.3f} est={est:.3f} Hz)")

        return {
            "ber": rx.ber, "ser": rx.ser,
            "n_bit_errors": rx.n_bit_errors,
            "snr_db": ch.snr_db, "ebn0_db": ebn0,
            "evm_pct": evm, "papr_db": papr,
            "estimated_doppler_hz": rx.estimated_doppler_hz,
            "status": "OK",
            "warnings": "; ".join(warnings) if warnings else "",
            "notes": "",
        }
    except Exception as exc:
        return {
            "ber":"N/A","ser":"N/A","n_bit_errors":"N/A",
            "snr_db":"N/A","ebn0_db":"N/A","evm_pct":"N/A",
            "papr_db":"N/A","estimated_doppler_hz":"N/A",
            "status":"ERROR",
            "warnings":"",
            "notes": str(exc)[:120],
        }

# ── CSV write helper ────────────────────────────────────────────────────────
def _csv(writer, group, name, mode, params, res):
    writer.writerow({
        "group": group, "scenario_name": name,
        "correction_mode": mode,
        "noise_temp_k":    params.get("noise_temp_k", ""),
        "hpa_ibo_db":      params.get("hpa_ibo_db", ""),
        "doppler_hz_true": params.get("doppler_hz", ""),
        "phase_noise_var": params.get("phase_noise_var", ""),
        "iq_amp_db":       params.get("iq_amp_db", ""),
        "iq_phase_deg":    params.get("iq_phase_deg", ""),
        "dc_offset_frac":  params.get("dc_offset_frac", ""),
        **{k: res[k] for k in [
            "ber","ser","n_bit_errors","snr_db","ebn0_db",
            "evm_pct","papr_db","estimated_doppler_hz",
            "status","warnings","notes"]},
    })

def _print_row(name, mode, res):
    ber   = f"{res['ber']:.4e}" if isinstance(res['ber'], float) else res['ber']
    evm   = f"{res['evm_pct']:.2f}%" if isinstance(res['evm_pct'], float) else ""
    edop  = (f"{res['estimated_doppler_hz']:.4f} Hz"
             if isinstance(res['estimated_doppler_hz'], float) else "")
    warn  = f"  ⚠  {res['warnings']}" if res['warnings'] else ""
    print(f"    [{mode:<7}] BER={ber}  EVM={evm}  est_Dop={edop}{warn}")

# ════════════════════════════════════════════════════════════════════════════
# GROUP A – Noise temperature
# ════════════════════════════════════════════════════════════════════════════
def sweep_noise_temperature(writer):
    group = "A_noise_temperature"
    temps = [0, 20, 290, 500, 1000, 5000]
    print(f"\n{'='*65}\n GROUP A: Noise Temperature Sweep\n{'='*65}")
    for t in temps:
        name = f"noise_T={t}K"
        print(f"  {name}")
        cfg = _base_cfg(); cfg.hpa_input_backoff_db = 30.0
        res = _run(cfg, override_noise_temp_k=t)
        _print_row(name, "none", res)
        _csv(writer, group, name, "none",
             {"noise_temp_k": t, "hpa_ibo_db": 30}, res)

# ════════════════════════════════════════════════════════════════════════════
# GROUP B – HPA back-off
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
# GROUP C – Doppler  (three modes: no_correction, ideal, blind)
# ════════════════════════════════════════════════════════════════════════════
def sweep_doppler(writer):
    """
    For each Doppler frequency run three cases:

    no_correction
        apply_doppler_correction = False  →  receiver ignores the offset.
        Demonstrates the BER floor caused by an uncompensated CFO.

    ideal
        apply_doppler_correction = True
        The TRUE Doppler value is passed directly to the receiver as
        override_doppler_hz.  This is the upper-bound / sanity check.
        It does NOT test the estimator; it tests the correction logic.

    blind
        apply_doppler_correction = True
        override_doppler_hz = None  →  receiver must estimate CFO from
        the 4th-power NDA estimator.  This is the realistic case.
        The estimated_doppler_hz column shows what the estimator found.
        A warning is emitted if the estimate differs from truth by > 10%.
    """
    group = "C_doppler"
    freqs = [0, 1, 3, 5, 10, 20]
    print(f"\n{'='*65}\n GROUP C: Doppler Frequency Sweep  (3 modes per value)\n{'='*65}")

    for f_d in freqs:
        print(f"  Doppler = {f_d} Hz")

        # ── no correction ─────────────────────────────────────────────
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_doppler_correction = False
        res_none = _run(cfg,
                        override_noise_temp_k=290.0,
                        override_doppler_hz=float(f_d))
        _print_row(f"Doppler={f_d}Hz", "none", res_none)
        _csv(writer, group, f"Doppler={f_d}Hz_no_correction", "none",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "doppler_hz": f_d},
             res_none)

        # ── ideal correction (true value passed to Rx) ────────────────
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_doppler_correction = True
        res_ideal = _run(cfg,
                         override_noise_temp_k=290.0,
                         override_doppler_hz=float(f_d),
                         rx_doppler_override=float(f_d),   # ← ideal
                         ref_evm_pct=(res_none['evm_pct']
                                      if isinstance(res_none['evm_pct'], float)
                                      else None))
        _print_row(f"Doppler={f_d}Hz", "ideal", res_ideal)
        _csv(writer, group, f"Doppler={f_d}Hz_ideal_correction", "ideal",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "doppler_hz": f_d},
             res_ideal)

        # ── blind correction (NDA estimator) ─────────────────────────
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_doppler_correction = True
        res_blind = _run(cfg,
                         override_noise_temp_k=290.0,
                         override_doppler_hz=float(f_d),
                         rx_doppler_override=None,          # ← blind
                         ref_evm_pct=(res_none['evm_pct']
                                      if isinstance(res_none['evm_pct'], float)
                                      else None))
        _print_row(f"Doppler={f_d}Hz", "blind", res_blind)
        _csv(writer, group, f"Doppler={f_d}Hz_blind_correction", "blind",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "doppler_hz": f_d},
             res_blind)

# ════════════════════════════════════════════════════════════════════════════
# GROUP D – Phase noise
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
# GROUP E – I/Q imbalance  (with and without correction)
# ════════════════════════════════════════════════════════════════════════════
def sweep_iq_imbalance(writer):
    group = "E_iq_imbalance"
    cases = [(0.0,0.0),(1.0,5.0),(2.0,10.0),(3.0,15.0),(5.0,20.0)]
    print(f"\n{'='*65}\n GROUP E: I/Q Imbalance Sweep\n{'='*65}")

    for amp_db, phase_deg in cases:
        label = (f"IQ_amp={amp_db}dB_phase={phase_deg}deg"
                 if amp_db > 0 or phase_deg > 0 else "IQ_none")
        print(f"  {label}")

        # baseline (no correction)
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db      = 30.0
        cfg.apply_iq_imbalance        = (amp_db > 0 or phase_deg > 0)
        cfg.iq_amplitude_imbalance_db = amp_db
        cfg.iq_phase_imbalance_deg    = phase_deg
        cfg.apply_iq_correction       = False
        res_none = _run(cfg, override_noise_temp_k=290.0)
        _print_row(label, "none", res_none)
        _csv(writer, group, f"{label}_no_correction", "none",
             {"noise_temp_k": 290, "hpa_ibo_db": 30,
              "iq_amp_db": amp_db, "iq_phase_deg": phase_deg},
             res_none)

        # with model-matched correction (uses fixed receiver.py)
        cfg2 = _base_cfg()
        cfg2.hpa_input_backoff_db      = 30.0
        cfg2.apply_iq_imbalance        = (amp_db > 0 or phase_deg > 0)
        cfg2.iq_amplitude_imbalance_db = amp_db
        cfg2.iq_phase_imbalance_deg    = phase_deg
        cfg2.apply_iq_correction       = True
        ref_evm = (res_none['evm_pct']
                   if isinstance(res_none['evm_pct'], float) else None)
        res_corr = _run(cfg2, override_noise_temp_k=290.0,
                        ref_evm_pct=ref_evm)
        _print_row(label, "blind", res_corr)
        _csv(writer, group, f"{label}_corrected", "blind",
             {"noise_temp_k": 290, "hpa_ibo_db": 30,
              "iq_amp_db": amp_db, "iq_phase_deg": phase_deg},
             res_corr)

# ════════════════════════════════════════════════════════════════════════════
# GROUP F – DC offset  (with and without correction)
# ════════════════════════════════════════════════════════════════════════════
def sweep_dc_offset(writer):
    group     = "F_dc_offset"
    fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    print(f"\n{'='*65}\n GROUP F: DC Offset Sweep\n{'='*65}")

    for frac in fractions:
        label = f"DC={frac:.2f}"
        print(f"  {label}")

        # no correction
        cfg = _base_cfg()
        cfg.hpa_input_backoff_db = 30.0
        cfg.apply_dc_offset      = (frac > 0)
        cfg.dc_offset_i          = frac
        cfg.dc_offset_q          = frac * 0.75
        cfg.apply_dc_correction  = False
        res_none = _run(cfg, override_noise_temp_k=290.0)
        _print_row(label, "none", res_none)
        _csv(writer, group, f"{label}_no_correction", "none",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "dc_offset_frac": frac},
             res_none)

        # with correction
        cfg2 = _base_cfg()
        cfg2.hpa_input_backoff_db = 30.0
        cfg2.apply_dc_offset      = (frac > 0)
        cfg2.dc_offset_i          = frac
        cfg2.dc_offset_q          = frac * 0.75
        cfg2.apply_dc_correction  = True
        ref_evm = (res_none['evm_pct']
                   if isinstance(res_none['evm_pct'], float) else None)
        res_corr = _run(cfg2, override_noise_temp_k=290.0,
                        ref_evm_pct=ref_evm)
        _print_row(label, "blind", res_corr)
        _csv(writer, group, f"{label}_corrected", "blind",
             {"noise_temp_k": 290, "hpa_ibo_db": 30, "dc_offset_frac": frac},
             res_corr)

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def main():
    start    = datetime.now()
    log_file = open(_LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print("=" * 65)
    print("  RF Satellite Link — External Scenario Runner  v2")
    print(f"  Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CSV     : {_CSV_PATH}")
    print(f"  Log     : {_LOG_PATH}")
    print("  Doppler correction modes: none / ideal / blind")
    print("  IQ correction: model-matched moment estimator (receiver.py fixed)")
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

    csv_file.flush(); csv_file.close()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*65}")
    print(f"  All scenarios complete.  Elapsed: {elapsed:.1f} s")
    print(f"  Results : {_CSV_PATH}")
    print(f"  Log     : {_LOG_PATH}")
    print("=" * 65)

    sys.stdout = sys.__stdout__
    log_file.flush(); log_file.close()
    print(f"\nDone.  Results saved to:\n  {_CSV_PATH}\n  {_LOG_PATH}\n")


if __name__ == "__main__":
    main()
