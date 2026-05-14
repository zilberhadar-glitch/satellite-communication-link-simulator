"""
temp_run_scenarios.py
---------------------
External scenario runner for the RF Satellite Link simulation.

PURPOSE
    Run a large matrix of parameter sweeps (noise temperature, HPA back-off,
    Doppler, phase noise, I/Q imbalance, DC offset) without touching any of the
    original project files.  All original files are imported read-only.

USAGE
    python temp_run_scenarios.py

OUTPUT FILES (written to the same folder as this script)
    terminal_scenario_results.csv   -- one row per run, all metrics
    terminal_scenario_log.txt       -- full console transcript

ORIGINAL FILES MODIFIED: NONE
"""

# ── standard library ────────────────────────────────────────────────────────
import copy
import csv
import math
import os
import sys
import io
import traceback
from datetime import datetime

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np

# ── redirect stdout so everything printed also lands in the log file ─────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH    = os.path.join(_SCRIPT_DIR, "terminal_scenario_log.txt")
_CSV_PATH    = os.path.join(_SCRIPT_DIR, "terminal_scenario_results.csv")

class _Tee(io.TextIOWrapper):
    """Write to both the real stdout and a log file simultaneously."""
    def __init__(self, real_stdout, log_file):
        self._real  = real_stdout
        self._log   = log_file
    def write(self, data):
        self._real.write(data)
        self._log.write(data)
        return len(data)
    def flush(self):
        self._real.flush()
        self._log.flush()

# ── CSV column headers ────────────────────────────────────────────────────────
CSV_FIELDS = [
    "group",
    "scenario_name",
    "noise_temp_k",
    "hpa_ibo_db",
    "doppler_hz",
    "doppler_correction",
    "phase_noise_var",
    "iq_amp_db",
    "iq_phase_deg",
    "iq_correction",
    "dc_offset_frac",
    "dc_correction",
    "ber",
    "ser",
    "n_bit_errors",
    "snr_db",
    "ebn0_db",
    "evm_pct",
    "papr_db",
    "estimated_doppler_hz",
    "status",
    "notes",
]

# ── helper: build a fresh Config with safe defaults ──────────────────────────
def _base_config():
    """
    Return a Config object with sensible defaults for external sweeps.
    Imports the unmodified config.py from the project.
    """
    from config import Config
    cfg = Config(
        modulation_order       = 16,
        carrier_freq_hz        = 4.0e9,
        sat_altitude_km        = 35_600.0,
        tx_antenna_diameter_m  = 0.4,
        rx_antenna_diameter_m  = 0.4,
        lna_gain_db            = 30.0,
        noise_temp_k           = 290.0,
        hpa_input_backoff_db   = 30.0,   # near-linear unless overridden
        doppler_hz             = 0.0,
        rolloff                = 0.25,
        span                   = 10,
        samples_per_symbol     = 8,
        num_symbols            = 10_000,
        random_seed            = 42,
        apply_doppler_correction = False,
        apply_phase_noise        = False,
        phase_noise_power_rad2   = 0.0,
        apply_iq_imbalance       = False,
        iq_amplitude_imbalance_db = 0.0,
        iq_phase_imbalance_deg    = 0.0,
        apply_iq_correction      = False,
        apply_dc_offset          = False,
        dc_offset_i              = 0.0,
        dc_offset_q              = 0.0,
        apply_dc_correction      = False,
        apply_agc                = True,
        agc_target_power         = 1.0,
        verbose                  = False,   # suppress per-step prints during sweep
    )
    return cfg


# ── core single-run function ──────────────────────────────────────────────────
def _run_one(cfg, override_noise_temp_k=None, override_doppler_hz=None):
    """
    Execute one full Tx → Channel → Rx simulation using the unmodified modules.

    Returns a dict with all metric fields, or a dict with status='ERROR'.
    """
    from transmitter import transmit
    from channel     import propagate
    from receiver    import receive, attach_srrc_h
    from metrics     import compute_ebn0_db, compute_evm, compute_papr_db
    from modulation  import bits_to_symbols

    try:
        # Cache the SRRC coefficients once (same pattern as main.py)
        attach_srrc_h(cfg)

        rng = np.random.default_rng(cfg.random_seed)

        # ── Transmitter ──────────────────────────────────────────────────────
        tx = transmit(cfg, rng)

        # ── Channel ──────────────────────────────────────────────────────────
        ch = propagate(
            tx.after_hpa, cfg, rng,
            override_doppler_hz   = override_doppler_hz,
            override_noise_temp_k = override_noise_temp_k,
        )

        # ── Receiver ────────────────────────────────────────────────────────
        # Pass the known Doppler to the receiver only when correction is on,
        # so it can use exact correction (mirrors main.py logic).
        rx = receive(
            ch.signal, tx.bits, cfg,
            override_doppler_hz = (
                override_doppler_hz if cfg.apply_doppler_correction else None
            ),
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        nt     = override_noise_temp_k if override_noise_temp_k is not None \
                 else cfg.noise_temp_k
        ebn0   = compute_ebn0_db(cfg, noise_temp_k=nt)
        tx_ref = bits_to_symbols(tx.bits, cfg.modulation_order)
        evm    = compute_evm(tx_ref, rx.symbols)
        papr   = compute_papr_db(tx.after_hpa)

        return {
            "ber":                  rx.ber,
            "ser":                  rx.ser,
            "n_bit_errors":         rx.n_bit_errors,
            "snr_db":               ch.snr_db,
            "ebn0_db":              ebn0,
            "evm_pct":              evm,
            "papr_db":              papr,
            "estimated_doppler_hz": rx.estimated_doppler_hz,
            "status":               "OK",
            "notes":                "",
        }

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"    [ERROR] {exc}")
        return {
            "ber": "N/A", "ser": "N/A", "n_bit_errors": "N/A",
            "snr_db": "N/A", "ebn0_db": "N/A", "evm_pct": "N/A",
            "papr_db": "N/A", "estimated_doppler_hz": "N/A",
            "status": "ERROR", "notes": str(exc)[:120],
        }


# ── printer for one result row ────────────────────────────────────────────────
def _print_row(name, res):
    ber   = f"{res['ber']:.4e}"   if isinstance(res['ber'],   float) else res['ber']
    snr   = f"{res['snr_db']:.1f}" if isinstance(res['snr_db'], float) else res['snr_db']
    evm   = f"{res['evm_pct']:.2f}%" if isinstance(res['evm_pct'], float) else res['evm_pct']
    edop  = (f"{res['estimated_doppler_hz']:.4f} Hz"
             if isinstance(res['estimated_doppler_hz'], float)
             else res['estimated_doppler_hz'])
    print(f"    BER={ber}  SNR={snr} dB  EVM={evm}  "
          f"est_Doppler={edop}  [{res['status']}]")


# ── CSV writer helper ────────────────────────────────────────────────────────
def _append_csv(writer, group, name, cfg_params, res):
    writer.writerow({
        "group":                group,
        "scenario_name":        name,
        "noise_temp_k":         cfg_params.get("noise_temp_k",  ""),
        "hpa_ibo_db":           cfg_params.get("hpa_ibo_db",    ""),
        "doppler_hz":           cfg_params.get("doppler_hz",    ""),
        "doppler_correction":   cfg_params.get("doppler_correction", ""),
        "phase_noise_var":      cfg_params.get("phase_noise_var", ""),
        "iq_amp_db":            cfg_params.get("iq_amp_db",     ""),
        "iq_phase_deg":         cfg_params.get("iq_phase_deg",  ""),
        "iq_correction":        cfg_params.get("iq_correction", ""),
        "dc_offset_frac":       cfg_params.get("dc_offset_frac",""),
        "dc_correction":        cfg_params.get("dc_correction", ""),
        **{k: res[k] for k in
           ["ber","ser","n_bit_errors","snr_db","ebn0_db",
            "evm_pct","papr_db","estimated_doppler_hz","status","notes"]},
    })


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO GROUPS
# ════════════════════════════════════════════════════════════════════════════

def sweep_noise_temperature(writer):
    """Group A – vary noise temperature, everything else nominal."""
    group = "A_noise_temperature"
    temps = [0, 20, 290, 500, 1000, 5000]
    print(f"\n{'='*65}")
    print(f" GROUP A: Noise Temperature Sweep")
    print(f"{'='*65}")
    for t in temps:
        name = f"noise_T={t}K"
        print(f"  {name}")
        cfg = _base_config()
        cfg.hpa_input_backoff_db = 30.0    # near-linear, isolate noise effect
        res = _run_one(cfg, override_noise_temp_k=t)
        _print_row(name, res)
        _append_csv(writer, group, name,
                    {"noise_temp_k": t, "hpa_ibo_db": 30,
                     "doppler_hz": 0, "doppler_correction": False},
                    res)


def sweep_hpa_backoff(writer):
    """Group B – vary HPA input back-off."""
    group = "B_hpa_backoff"
    ibos  = [30, 15, 10, 7, 5, 3, 1]
    print(f"\n{'='*65}")
    print(f" GROUP B: HPA Input Back-Off Sweep")
    print(f"{'='*65}")
    for ibo in ibos:
        name = f"HPA_IBO={ibo}dB"
        print(f"  {name}")
        cfg = _base_config()
        cfg.hpa_input_backoff_db = ibo
        res = _run_one(cfg, override_noise_temp_k=290.0)
        _print_row(name, res)
        _append_csv(writer, group, name,
                    {"noise_temp_k": 290, "hpa_ibo_db": ibo,
                     "doppler_hz": 0, "doppler_correction": False},
                    res)


def sweep_doppler(writer):
    """Group C – vary Doppler, with and without correction."""
    group  = "C_doppler"
    freqs  = [0, 1, 3, 5, 10, 20]
    print(f"\n{'='*65}")
    print(f" GROUP C: Doppler Frequency Sweep")
    print(f"{'='*65}")
    for f_d in freqs:
        for correct in (False, True):
            tag  = "corrected" if correct else "no_correction"
            name = f"Doppler={f_d}Hz_{tag}"
            print(f"  {name}")
            cfg = _base_config()
            cfg.hpa_input_backoff_db  = 30.0
            cfg.apply_doppler_correction = correct
            res = _run_one(cfg,
                           override_noise_temp_k = 290.0,
                           override_doppler_hz   = float(f_d))
            _print_row(name, res)
            _append_csv(writer, group, name,
                        {"noise_temp_k": 290, "hpa_ibo_db": 30,
                         "doppler_hz": f_d,
                         "doppler_correction": correct},
                        res)


def sweep_phase_noise(writer):
    """Group D – vary phase noise variance."""
    group   = "D_phase_noise"
    # 0 means disabled; the rest are σ² in rad²
    variances = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    print(f"\n{'='*65}")
    print(f" GROUP D: Phase Noise Variance Sweep")
    print(f"{'='*65}")
    for var in variances:
        name = f"phase_noise_var={var:.0e}" if var > 0 else "phase_noise_var=0"
        print(f"  {name}")
        cfg = _base_config()
        cfg.hpa_input_backoff_db  = 30.0
        cfg.apply_phase_noise     = (var > 0)
        cfg.phase_noise_power_rad2 = var
        res = _run_one(cfg, override_noise_temp_k=290.0)
        _print_row(name, res)
        _append_csv(writer, group, name,
                    {"noise_temp_k": 290, "hpa_ibo_db": 30,
                     "doppler_hz": 0, "phase_noise_var": var},
                    res)


def sweep_iq_imbalance(writer):
    """Group E – vary I/Q amplitude + phase imbalance, with and without correction."""
    group = "E_iq_imbalance"
    cases = [
        (0.0,  0.0),
        (1.0,  5.0),
        (2.0, 10.0),
        (3.0, 15.0),
        (5.0, 20.0),
    ]
    print(f"\n{'='*65}")
    print(f" GROUP E: I/Q Imbalance Sweep")
    print(f"{'='*65}")
    for amp_db, phase_deg in cases:
        for correct in (False, True):
            tag  = "corrected" if correct else "no_correction"
            name = (f"IQ_amp={amp_db}dB_phase={phase_deg}deg_{tag}"
                    if amp_db > 0 or phase_deg > 0
                    else f"IQ_none_{tag}")
            print(f"  {name}")
            cfg = _base_config()
            cfg.hpa_input_backoff_db      = 30.0
            cfg.apply_iq_imbalance        = (amp_db > 0 or phase_deg > 0)
            cfg.iq_amplitude_imbalance_db = amp_db
            cfg.iq_phase_imbalance_deg    = phase_deg
            cfg.apply_iq_correction       = correct
            res = _run_one(cfg, override_noise_temp_k=290.0)
            _print_row(name, res)
            _append_csv(writer, group, name,
                        {"noise_temp_k": 290, "hpa_ibo_db": 30,
                         "iq_amp_db": amp_db, "iq_phase_deg": phase_deg,
                         "iq_correction": correct},
                        res)


def sweep_dc_offset(writer):
    """Group F – vary DC offset fraction, with and without correction."""
    group    = "F_dc_offset"
    # DC offset as a fraction of signal RMS (matches the channel.py scaling)
    fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    print(f"\n{'='*65}")
    print(f" GROUP F: DC Offset Sweep")
    print(f"{'='*65}")
    for frac in fractions:
        for correct in (False, True):
            tag  = "corrected" if correct else "no_correction"
            name = (f"DC={frac:.2f}_{tag}"
                    if frac > 0 else f"DC=0_{tag}")
            print(f"  {name}")
            cfg = _base_config()
            cfg.hpa_input_backoff_db = 30.0
            cfg.apply_dc_offset      = (frac > 0)
            cfg.dc_offset_i          = frac
            cfg.dc_offset_q          = frac * 0.75    # slight asymmetry
            cfg.apply_dc_correction  = correct
            res = _run_one(cfg, override_noise_temp_k=290.0)
            _print_row(name, res)
            _append_csv(writer, group, name,
                        {"noise_temp_k": 290, "hpa_ibo_db": 30,
                         "dc_offset_frac": frac, "dc_correction": correct},
                        res)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    start = datetime.now()

    # Open log file and tee stdout into it
    log_file = open(_LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print("=" * 65)
    print("  RF Satellite Link — External Scenario Runner")
    print(f"  Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CSV     : {_CSV_PATH}")
    print(f"  Log     : {_LOG_PATH}")
    print("  Original project files: NOT MODIFIED")
    print("=" * 65)

    # Open CSV
    csv_file = open(_CSV_PATH, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    # ── Run all groups ────────────────────────────────────────────────────
    sweep_noise_temperature(writer)
    sweep_hpa_backoff(writer)
    sweep_doppler(writer)
    sweep_phase_noise(writer)
    sweep_iq_imbalance(writer)
    sweep_dc_offset(writer)

    # Close CSV
    csv_file.flush()
    csv_file.close()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*65}")
    print(f"  All scenarios complete.")
    print(f"  Elapsed : {elapsed:.1f} s")
    print(f"  Results : {_CSV_PATH}")
    print(f"  Log     : {_LOG_PATH}")
    print("=" * 65)

    # Restore stdout and close log
    sys.stdout = sys.__stdout__
    log_file.flush()
    log_file.close()

    # Print final message directly (after restoring stdout)
    print(f"\nDone.  Results saved to:\n  {_CSV_PATH}\n  {_LOG_PATH}\n")


if __name__ == "__main__":
    main()
