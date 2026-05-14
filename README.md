# RF Satellite Link Simulation – Python (MATLAB-Equivalent)
### Faithful Python reproduction of the MATLAB/Simulink RF Satellite Link example
**Reference:** https://www.mathworks.com/help/comm/ug/rf-satellite-link.html

---

```
satellite_link/
├── config.py                      All parameters + link-budget (updated)
├── modulation.py                  Gray-coded 16-QAM mapper / demapper
├── filters.py                     SRRC filter design & application
├── impairments.py                 HPA, DPD, path loss, Doppler, AWGN,
│                                  colored phase noise, I/Q imbalance,
│                                  absolute DC offset, LNA gain (updated)
├── transmitter.py                 Tx chain: bits→symbols→SRRC→DPD→HPA (updated)
├── channel.py                     Downlink: path loss→Doppler→noise→LNA→... (updated)
├── receiver.py                    Rx chain: MF→CFO(PLL/blind/ideal)→IQ→demod (updated)
├── metrics.py                     BER theory, EVM, PAPR, link budget
├── plots.py                       Constellation, spectrum, BER, HPA AM/AM
├── main.py                        18-scenario runner (updated)
├── run_matlab_equivalence_tests.py MATLAB-equivalence test suite (new)
└── requirements.txt               numpy, scipy, matplotlib
```

## Quick start
```bash
pip install -r requirements.txt
python main.py                           # 18 scenarios + plots
python run_matlab_equivalence_tests.py  # full MATLAB-equivalence suite
```

PowerShell:
```powershell
python main.py
python run_matlab_equivalence_tests.py
```

---

## What changed vs the original Python

### 1. Digital Pre-Distortion (DPD) — NEW
**MATLAB equivalent:** DPD subsystem in the RF Satellite Link Simulink model.

`impairments.saleh_dpd()` implements an analytic LUT-based inverse of the Saleh
AM/AM and AM/PM functions.  It maps the desired output amplitude to the required
pre-distorted input amplitude by inverting the AM/AM curve over its monotone
region [0, r_sat], and subtracts the corresponding AM/PM phase shift.

Enable with `cfg.apply_dpd = True`.  Must be combined with `cfg.apply_hpa = True`.

Expected outcome: IBO = 7 dB + DPD gives BER close to the near-linear IBO = 30 dB case.

---

### 2. Block ordering: noise BEFORE LNA — FIXED
**MATLAB block diagram order:**
```
Rx antenna → Thermal Noise → LNA → Phase Noise → I/Q → DC → Rx processing
```
Original Python applied LNA *before* noise.  `channel.py` now matches MATLAB.

SNR note: because SNR = P_signal / P_noise and the LNA multiplies both by the same
gain, the SNR value is physically identical to the original.  The block order matters
for fidelity of the hardware model, not for the SNR number.

---

### 3. Default noise temperature: 290 K → 20 K — FIXED
`config.py` default `noise_temp_k = 20.0 K` (MATLAB nominal).
290 K and 500 K are still tested as explicit scenarios.

---

### 4. Colored phase noise — NEW
**MATLAB equivalent:** `comm.PhaseNoise` / Phase Noise block.

`impairments.add_colored_phase_noise()` shapes white noise with a 1/f² PSD:

    S_phi(f) = L0 × (f0/f)²    [rad²/Hz]

Specified as `phase_noise_dbc_hz` (dBc/Hz at `phase_noise_freq_offset_hz`).
Default: −85 dBc/Hz at 100 Hz.

White phase noise (`phase_noise_use_white = True`) is kept as a legacy option.

**Remaining difference:** MATLAB's block accepts a full PSD table; this implementation
uses a single-point 1/f² approximation.  For exact reproduction, a full table lookup
filter would be required.

---

### 5. DC offset: absolute mode — FIXED
`cfg.dc_offset_mode = "absolute"` uses MATLAB-compatible absolute offsets
(`dc_offset_i_abs = 1e-8`, `dc_offset_q_abs = 5e-8`).
`"relative"` mode (fraction of RMS) kept as legacy.

---

### 6. Separate I/Q amplitude-only / phase-only scenarios — ADDED
New MATLAB-compatible values: 3 dB amplitude imbalance, 20° phase imbalance.
Scenarios 13–16 in `main.py` test each in isolation, with and without correction.

---

### 7. Carrier synchroniser (PLL) — NEW
`cfg.cfo_correction_mode = "carrier_sync"` activates a decision-directed
2nd-order PLL in `receiver.py`:
- Proportional-plus-integral (PI) loop filter
- Loop bandwidth `carrier_sync_loop_bw` (default 0.01 normalised)
- Damping factor `carrier_sync_damping` (default 0.707 = Butterworth)

**Comparison of modes:**

| Mode | Description | MATLAB equivalent |
|------|-------------|-------------------|
| `"ideal"` | True CFO applied directly | Ideal correction / test mode |
| `"blind"` | 4th-power NDA batch FFT | Open-loop estimator |
| `"carrier_sync"` | Symbol-by-symbol DD-PLL | `comm.CarrierSynchronizer` |

**Remaining difference:** MATLAB's `comm.CarrierSynchronizer` uses a proprietary
loop filter implementation; the Python PLL matches the standard 2nd-order
Gardner/DD-PLL structure but may differ in transient behaviour.

---

### 8. HPA bypass — ADDED
`cfg.apply_hpa = False` bypasses the Saleh TWTA entirely (ideal linear amplifier).
MATLAB equivalent: disconnecting the HPA block in the Simulink model.

---

### 9. Physical symbol rate support — ADDED
Set `cfg.symbol_rate_baud > 0` to use physical units.
`sample_rate_hz = symbol_rate_baud × samples_per_symbol`.
Default `symbol_rate_baud = 0` keeps normalised mode (1 sym/s, backward compatible).

---

## Simulation chain (updated)

```
[Random bits]
     ↓ bits_to_symbols()
[16-QAM symbols]  ──────────────► constellation "Before HPA"
     ↓ tx_filter()  [SRRC ×8]
[Oversampled waveform]
     ↓ saleh_dpd()  [optional – LUT-based inverse Saleh]
[Pre-distorted waveform]
     ↓ saleh_hpa()  [Saleh TWTA]  OR  bypass
[After-HPA waveform]  ──────────► constellation "After HPA"
     ↓ apply_path_loss()
     ↓ apply_doppler()
     ↓ add_awgn_noise()           ← BEFORE LNA (MATLAB order)
     ↓ apply_lna_gain()           ← AFTER noise (MATLAB order)
     ↓ add_colored_phase_noise()  [optional – 1/f² model]
     ↓ apply_iq_imbalance()       [optional – amp or phase or both]
     ↓ add_dc_offset()            [optional – absolute or relative]
     ↓ ─ Receiver ──────────────────────────────────────────────
     ↓ SRRC matched filter + downsample
     ↓ CFO correction:
     │    "ideal"        → multiply by exp(-j2π·f_true·t)
     │    "blind"        → 4th-power NDA FFT estimator
     │    "carrier_sync" → decision-directed 2nd-order PLL  ← MATLAB-like
     ↓ data-aided residual phase correction  (ideal/blind only)
     ↓ DC correction  (subtract mean)
     ↓ I/Q imbalance correction  (2nd-order statistics)
     ↓ AGC
     ↓ symbols_to_bits()  [nearest-neighbour]
[Decoded bits] ──────────────────► BER / SER
```

---

## Scenarios & expected results (main.py)

| # | Scenario | Expected BER |
|---|----------|-------------|
| 1 | Clean (bypass, no noise) | 0 |
| 2 | T=20 K (MATLAB nominal) | 0 |
| 3 | T=290 K | 0 |
| 4 | T=500 K | 0 |
| 5 | Doppler 3 Hz, no correction | ~50% |
| 6 | Doppler 3 Hz, blind NDA | ~0 |
| 7 | Doppler 3 Hz, carrier_sync PLL | ~0 |
| 8 | HPA bypass | 0 |
| 9 | HPA IBO=30 dB | 0 |
| 10 | HPA IBO=7 dB | ~7% |
| 11 | HPA IBO=7 dB + DPD | ≪7% |
| 12 | HPA IBO=1 dB | ~18% |
| 13 | I/Q amp-only 3 dB, no corr | low |
| 14 | I/Q amp-only 3 dB, corrected | 0 |
| 15 | I/Q phase-only 20°, no corr | low |
| 16 | I/Q phase-only 20°, corrected | 0 |
| 17 | Colored phase noise −85 dBc/Hz | 0 |
| 18 | DC offset absolute, corrected | 0 |

---

## MATLAB-vs-Python comparison (after fixes)

| Feature | MATLAB | Python (after fixes) | Fidelity |
|---------|--------|----------------------|----------|
| Default noise temp | 20 K | 20 K ✓ | Exact |
| Noise / LNA order | Noise → LNA | Noise → LNA ✓ | Exact |
| HPA model | Saleh TWTA | Saleh TWTA ✓ | Exact |
| HPA bypass | Yes | Yes ✓ | Exact |
| DPD | LUT-based inverse | LUT-based inverse ✓ | Close |
| Phase noise | PSD table + filter | 1/f² single-point approx | Approximate |
| Carrier sync | `comm.CarrierSynchronizer` (DD-PLL) | DD 2nd-order PLL ✓ | Close |
| DC offset | Absolute (1e-8/5e-8) | Absolute ✓ | Exact |
| I/Q imbalance | 3 dB / 20° separate | 3 dB / 20° separate ✓ | Exact |
| Symbol rate | Physical (Baud) | Physical or normalised ✓ | Exact |
| Modulation | 16-QAM Gray | 16-QAM Gray ✓ | Exact |
| SRRC filter | Raised cosine pair | SRRC pair ✓ | Exact |

### Remaining differences (cannot be exactly reproduced without MATLAB internals)
1. **Phase noise PSD shape** – MATLAB uses an interpolated table; Python uses 1/f².
2. **PLL transient convergence** – `comm.CarrierSynchronizer` uses a proprietary
   normalisation; the Python PLL may converge slower on short bursts.
3. **Random number generator** – MATLAB uses its own RNG; sequences differ.
4. **Block internal states** – Simulink blocks carry inter-frame state; Python
   processes each burst independently.

---

## Running the equivalence suite (PowerShell)

```powershell
cd path\to\satellite_link
pip install -r requirements.txt
python run_matlab_equivalence_tests.py
# Results: matlab_equivalence_outputs\matlab_equivalence_results.csv
#          matlab_equivalence_outputs\matlab_equivalence_log.txt
```
