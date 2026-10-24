# GlucoTwin

**Glucose forecasting with a Reliability Lab, inspectable forecast receipts and measured handling of missing or delayed sensor inputs.** Built for Digital Twin Challenge 2026.

The patient state combines a linked baseline clinical profile with incoming glucose and activity observations. A real incremental function issues six forecasts, at 5-minute intervals through 30 minutes. A clinician-facing dashboard shows issued forecasts, later outcomes, data loss and held-out evidence.

## Team and college information

**Team TwinPulse**

| Member | Role | Program | College |
|---|---|---|---|
| Rimzim Kumari | Team Leader | Computer Science and Engineering (CSE) | VIT Bhopal |
| Aksh Agrawal | Team Member | Electronics and Communication Engineering (ECE) | VIT Vellore |

## Problem statement

A current glucose reading describes the present but does not directly estimate the next half hour. We investigate whether past glucose changes, linked clinical measurements and available activity can improve short-horizon forecasting over simple baselines. A practical system must also show missing evidence, errors and warning burden.

## Healthcare use case

The prototype supports a researcher or clinician reviewing anonymous CGM recordings, with a focus on Type 2 diabetes. They can inspect the next 30-minute trajectory, compare it with later observations, and review successful, missed and false warnings. Better-informed review is the intended benefit. Prospective clinical benefit has not been demonstrated. The prototype offers no dosing or treatment recommendations.

## Submission files

- [Architecture diagram (PDF)](output/pdf/GlucoTwin_Reliability_Architecture.pdf)
- [Project presentation and outcomes (editable PowerPoint)](output/presentation/GlucoTwin_TwinPulse_Reliability_Edition.pptx)
- [Complete project and results (PDF)](output/pdf/GlucoTwin_Project_Reliability_Edition.pdf)
- [Submission checklist](docs/SUBMISSION_CHECKLIST.md)
- [Video timing and recording handoff](docs/VIDEO_HANDOFF.md)

## Public GitHub repository

[Team TwinPulse submission repository](https://github.com/rimzim9934/glucotwin-digital-twin)

This repository contains the GlucoTwin Reliability Lab submission. Architecture, presentation, source code and evaluation evidence are linked above. The video link is pending.

## Demo video

**Pending: the team is preparing the recording and its unlisted YouTube link.** No video has been uploaded or verified by this build. Replace this notice with the actual unlisted YouTube URL before submission.

The latest supplied form explicitly asks for **15-20 minutes**. Earlier materials asked for a minimum of 20 minutes. Aim for **20:00 final runtime** to meet both wordings, or confirm the organizer's latest rule if an exact runtime is impractical. The earlier 24-minute guide is an extended narration draft and must be shortened for this form.

## Technical stack

Python 3.11, NumPy 1.26.4, pandas 2.0.3, SciPy 1.14.0, scikit-learn 1.3.0, joblib 1.3.2, Streamlit 1.52.2 and Plotly 5.17.0. Dependencies are pinned in [requirements.txt](requirements.txt). The numeric-data downloader uses Python's standard HTTP/ZIP libraries and certifi. The demo runs locally on CPU.

## AI/ML model and framework details

scikit-learn implements the six compared approaches: persistence, capped recent trend, ridge regression, boosted CGM, clinical plus CGM, and clinical plus CGM plus activity. Thirteen past glucose points span an hour. Past-only trend/variability and time-of-day features feed six delta regressors for +5 through +30 minutes. Fusion adds linked age, BMI, HbA1c and fasting glucose, with activity summaries in the activity model.

The tree models use `MultiOutputRegressor(HistGradientBoostingRegressor)` with 100 iterations, up to 15 leaves, minimum 80 samples per leaf, learning rate 0.06, L2 regularization 10 and seed 20261019. Training weights balance participant contributions. Ridge uses train-fitted median imputation/scaling and alpha 100. Validation selected **boosted CGM**. Bias-update rates 0, 0.03 and 0.1 were tested, and zero won. Empirical error bands come from validation residuals. The model does not output calibrated event probabilities.

## Run locally

Python 3.11 was used. From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py
python3 prepare.py
python3 train.py dev
python3 train.py final --verify
python3 scripts/check.py
python3 run.py
```

`final --verify` regenerates predictions and checks them against the committed evaluation. For a new experiment without recorded results, use `final` once. Do not tune on these test participants and continue calling the result independent. The download retrieves only numeric members from the official version-pinned ZIP, skips meal photos, verifies ZIP CRC32 and records member SHA256. It does **not** verify the whole-archive checksum. The two dictionaries match official SHA256 values.

Local dashboard: http://127.0.0.1:8502/ . This repository contains GlucoTwin only.

## Results, including failures

27 train / 9 validation / 9 reserved test participants. Models were frozen before the final test. One test participant has no supported forecast windows under the cadence rule; 8 have error measurements. All 9 remain in monitoring coverage. Forecast-error comparison uses the same 18,746 timestamps, equally averaged by patient. Initial 24 hours are excluded.

| Model | All-cohort MAE, mg/dL | Type 2 MAE, mg/dL |
|---|---:|---:|
| Unchanged glucose | 14.21 | 16.43 |
| Recent trend | 19.80 | 20.30 |
| Regularized linear CGM | 13.19 | 14.51 |
| Boosted CGM, validation-selected | 12.64 | 14.12 |
| Clinical + CGM | 12.53 | 14.03 |
| Clinical + CGM + activity | 12.47 | 14.04 |

The validation-selected model reduced error by **11.1%** versus unchanged glucose on this reserved test. Clinical fusion is implemented and performs well against persistence, but did not outperform CGM-only on validation; its small test advantage does not establish reliable added value. Online bias adaptation was worse on validation and is disabled. The twin still maintains a changing history/state; we do not claim successful learned personalization.

### Corrected warning evaluation (scoring v2)

A research event is a new crossing strictly above 180 mg/dL sustained across four consecutive five-minute samples (15 minutes). Current high readings suppress new-onset warnings. The selected model's forecast path reaching 170 mg/dL triggers its criterion, with a 30-minute cooldown. Thresholds remain frozen from the original validation run.

The scorer now emits warnings and advances cooldown using present evidence only. It credits an observed sustained positive onset even when a later gap truncates follow-up. Unmatched alerts without complete 45-minute follow-up are explicitly **censored**, with unknown outcomes. They do not count as false warnings.

- Selected model: **162/163** eligible onsets credited, **1** genuine missed recrossing, median credited lead **10 minutes**. **268 false warnings / 1,157.17 eligible non-event hours = 0.232/hour**. **3** emitted warnings have censored outcomes.
- Persistence: **139/163**, 246 false warnings, 0.213/hour. Catching more with more warnings is not fixed-false-alarm superiority.
- Type 2 subset: **106/106** credited, 161 false warnings, **0.466/hour**, 1 censored warning. This subset includes only **3 test participants**. It does not establish perfect performance or prospective benefit.
- Monitoring coverage: **78.8%** overall including the unsupported participant, **93.7%** in Type 2.
- Nominal 90% residual band: **86.5%** patient-mean test coverage, **50.8 mg/dL** width at +30 minutes.

The previous 161/163 and 105/106 counts came from the flawed future-coverage gate. [Scoring-v2 audit](reports/scoring_v2_audit.json) records all model comparisons. Historical outputs remain in `reports/scoring_v1/`. No models were retrained and no thresholds were tuned during this correction. Warning sensitivity is not overall prediction accuracy. The correction uses inspected test records and requires independent replication.

## Forecast Reliability Lab

The new dashboard tab compares the same chronological recording and frozen engine under original delivery, repeated signal loss and delayed packets. It exposes input timestamps, reasons for withholding, recovery, and downloadable JSON forecast receipts.

| Delivery condition | Patient-mean forecast coverage | MAE on retained targets | Normal MAE on the same targets |
|---|---:|---:|---:|
| Original observations | 78.8% | 12.64 | 12.64 |
| 15-minute loss every 6 hours | 62.4% | 12.60 | 12.60 |
| 10-minute delay for 30 minutes every 6 hours | 59.2% | 12.59 | 12.59 |

MAE units: mg/dL at +30 minutes. Coverage includes all 9 reserved participants. Eight contribute forecast errors. Missing/delayed schedules depend only on elapsed time. Packets older than 5 minutes fail the freshness rule, retain their source timestamps, and never replace the missing current observation or retroactively backfill a gap.

The guarded system withholds **3,909** additional opportunities under loss and **4,680** under delay, rejecting **2,213** stale packets. Retained forecasts are identical to normal because their inputs remain unchanged. Scenario MAEs use different retained targets, so the small numerical decrease is **not an accuracy gain**. This engineering stress test measures availability, withholding and recovery on already inspected recordings, not physiological robustness or new clinical validation.

See the [full reliability benchmark](reports/reliability_benchmark.json), [per-patient results](reports/reliability_per_patient.csv) and [Reliability Lab protocol](docs/RELIABILITY_LAB.md).

### Forecast receipts and recent observed errors

A receipt preserves the received input window, model-artifact SHA256, clinical-profile usage, issuance time and six predictions. Its issued payload has a content digest. Actual outcomes append separately when target readings arrive. The digest supports content checking, not authentication or a signed medical record. [Example receipt from P042](output/evidence/example_forecast_receipt.json).

Patient twin displays errors for the most recent 20 observed matured +30-minute forecasts in that replay. Future targets and missing outcomes contribute no error value. These overlapping local errors are descriptive, not calibrated confidence, event probability or a safety guarantee.

## Data and important limitation

CGMacros v1.0.0 contains 45 linked participants: 14 Type 2, 16 prediabetes and 15 without diabetes. We downloaded 48 numeric CSV members. The audit found 687,580 source rows and 141,262 canonical five-minute grid rows.

**The released minute-level CGM stream is interpolated.** We infer the dominant five-minute phase from integer-valued readings in the first 24 hours, retain that phase thereafter, reject fractional/out-of-range values and never fill gaps. This reconstructs cadence but cannot fully recover original native timestamps or hidden source interpolation across missing intervals. Results are provisional reconstructed-cadence research findings; independent raw-CGM validation is still required. Participant 18 changes phase after a large gap and has no supported post-initialization windows under the fixed rule.

Clinical fields: age, BMI, HbA1c, fasting glucose. HbA1c percent interpretation follows source cohort definitions and values, despite a contradictory dictionary unit. MET fields are divided by 10. Activity calories are distinct from meal calories; participant 7 lacks the activity-calorie column and is handled as missing. Meals, Libre, fingerprints/IDs, future observations and medication guesses are not features. Fasting insulin is not administered insulin.

The cohort was recruited in the US. No Indian-population, Indian-diet, clinical-benefit, treatment-effect or device-deployment claims are supported. A real deployment would need CGM connectivity; an ordinary smartwatch alone does not supply glucose.

## Architecture

- `prepare.py`: participant linkage, source audit, first-day cadence initialization, past-only features and grouped splits.
- `reliability.py`: causal input/forecast ledger, controlled delivery schedules, source timestamp checks and receipt serialization.
- `core.py`: model wrapper and `TwinState.update(observation)`. Receives one canonical observation; retains history, pending forecasts, missingness and optional matured-error bias.
- `train.py`: six baselines/ablations; validation-only model, interval and warning-policy selection; reserved test evaluation and reproducibility check.
- `evaluation.py`: offline future labels, episode matching, cooldown and censoring. Never used by the inference engine.
- `app.py`: Streamlit + Plotly research dashboard; actual incremental inference with hidden-future/reveal controls, sensor-loss simulation and evidence views.
- `tests/`: causality, missingness/recovery, event boundaries, cooldown, split isolation, feature equivalence and trained-model offline/incremental parity.
- `reports/`: source manifest, cohort audit, fixed splits/config, all validation candidates, complete test results and per-patient measurements.

No cloud service, account credentials or physical sensor is required to run the research replay. Model files are local artifacts and must only be loaded from trusted sources.

## Demo

Select a patient and engine. Start with future outcomes hidden; advance by 5 or 30 minutes. Reveal later readings to compare issued forecasts. Review a successful warning, missed event and false warning. Scenario selection uses offline annotations for demonstration; inference only receives observations at or before the cursor. Activate signal loss, advance, then deactivate it and accumulate 60 minutes of valid readings to restore support.

The dashboard begins a fresh replay state at each selected scenario. The frozen deployment uses no bias adaptation, so the population forecasts match offline inference once sufficient history is supplied. Scenario jumps can also reset warning cooldown. A selected screen is not an exact full-history alert-policy reproduction. Scoring v2 corrects causal emission and outcome censoring as described above. Offline evaluation uses future labels only to judge already emitted warnings.

## License and citation

Original code: MIT, see `LICENSE`. Dataset: **CC BY-NC-SA 4.0**, separate from code licensing. Do not relabel original records or derived dataset artifacts as MIT. Models are not included in the source repository; review applicable data/model redistribution terms before sharing trained artifacts. Follow the download procedure instead of publishing the archive.

Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). *CGMacros: a scientific dataset for personalized nutrition and diet monitoring*, version 1.0.0. PhysioNet. https://doi.org/10.13026/3z8q-x658

Official dataset: https://physionet.org/content/cgmacros/1.0.0/

## Submission status

Local code, team/college details, README requirements, licenses, updated architecture PDF and reliability-edition presentation PPTX are ready. Sixteen checks pass, including receipts, stale-input recovery and scorer regressions. Frozen forecast and scoring-v2 reproduction matches the current reports. The [checklist](docs/SUBMISSION_CHECKLIST.md) records the evidence and remaining items.

**Still pending:** the team's unlisted YouTube video and its README link. Before final submission, check that the repository and all linked files open while signed out. Only the team leader should complete the organizer form.

The earlier supplied deadline was 20 October 2026, 19:00 IST, which is in the past as of this checklist audit. These screenshots do not establish a new deadline. Confirm the current submission window on the organizer platform. Only the team leader should make the final submission.
