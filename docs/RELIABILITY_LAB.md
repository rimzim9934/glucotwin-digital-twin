# Forecast Reliability Lab protocol - 24 October 2026

## Scope

Engineering stress tests and inspection on the existing, previously examined reserved recordings. This is not an independent test of improved model accuracy or clinical reliability. No training, model selection, threshold tuning or bias adaptation occurs.

## Controlled observations

The first 24 hours still establish the original fixed cadence. After initialization, schedules use five-minute elapsed-time indices modulo 72 (six hours). Signal loss removes indices 36-38 of each cycle (15 minutes). Delayed delivery covers 36-41 (30 minutes). A delayed packet retains its measurement timestamp from two grid steps earlier, ten minutes of age. It fails the five-minute freshness threshold. The current input becomes unavailable. No old packet receives the current timestamp, replaces the missing reading, or fills history retroactively. No future observations or event labels choose the disturbance schedule.

A supported forecast requires 13 consecutive fresh, finite glucose points spanning 60 minutes. After a disruption ends, the next 13 fresh points restore support. The same frozen engine runs on identical retained input windows. The offline full-cohort benchmark therefore reuses the saved predictions with the causal support mask. Tests compare incremental normal and disrupted runs on all retained targets and verify future isolation.

## Comparison

Coverage uses every post-initialization grid row and averages participants equally, including P018 with no supported forecasts. MAE uses observed +30 targets among retained supported forecasts. Compare each guarded scenario against normal on those exact retained targets. Equal paired errors show no accuracy gain. The headline result is reduced availability: 78.8% original, 62.4% loss, 59.2% delay. Scheduled drops on already missing data create no additional scored target.

## Receipts

The fixed payload includes received input values/times, profile usage, model-artifact SHA256, issued timestamp, target timestamps, predictions, and recent matured-error evidence at issuance. A digest covers that payload. Later observations update only the outcomes outside the hashed payload. An absent target stays unobserved. An example JSON is in output/evidence/.

The content digest is not a signature. These are research records, not authenticated clinical records. The model-artifact hash identifies a local binary, not a guarantee of model quality. Clinical profiles accompany CGM-only receipts with clinical_profile_used=false.

## Recent observed performance

Only an actually received, finite target at its target timestamp matures an error. The UI averages the latest 20 observed +30-minute errors. Forecasts overlap. This statistic describes a replay's recent behavior and is neither calibrated confidence nor a safety score. It begins unavailable and never uses future targets.

## Reproduction

Run python3 scripts/benchmark_reliability.py and python3 scripts/check.py from glucotwin. The dashboard runs actual incremental inference for a 6/12/24-hour recording segment under each condition. Original shifted research dates remain visible.
