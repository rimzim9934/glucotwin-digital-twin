# GlucoTwin status - 24 October 2026, reliability upgrade

Complete locally: frozen forecast pipeline, corrected scoring-v2 accounting, Forecast Reliability Lab with three delivery scenarios, causal forecast receipts, recent matured error, updated project report/architecture/presentation, and 16 passing checks. Frozen forecast/evaluation reproduction matches the saved updated results. Models and thresholds remain unchanged.

Selected model: 12.64 vs 14.21 mg/dL MAE (11.1% lower). Scoring v2: 162/163 credited onsets, 268 false warnings, 3 censored alerts. Type 2 subset: only 3 participants, 106/106 credited with 0.466 false warnings/hour and 1 censored emission. These counts are not overall accuracy or clinical validation.

Disruption benchmark: forecast coverage 78.8% original, 62.4% repeated loss, 59.2% delayed packets. Retained forecasts match normal exactly on paired targets. No accuracy gain is claimed.

The complete project is published at https://github.com/rimzim9934/glucotwin-digital-twin on main. Anonymous download checks passed for the README and all linked files. The unlisted video/link and final leader submission remain pending. See SUBMISSION_CHECKLIST.md and reports/github_publication_verification.json.

Launch: python3 glucotwin/run.py. Open http://127.0.0.1:8502/ and select Reliability lab.
