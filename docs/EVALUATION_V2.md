# Event-scoring correction - 24 October 2026

The original docs/PROTOCOL.md and frozen_config.json remain unchanged to preserve experiment fingerprints. This amendment corrects implementation of the alert-policy accounting, without retuning its operating point.

An alert is emitted when input is supported, initialization is complete, current glucose is at most 180, forecast peak meets the frozen threshold, and the previous emitted alert is at least 30 minutes old. Every emission advances cooldown, including an alert whose future outcome cannot later be judged.

A fully observed sustained onset within the next 30 minutes credits a warning even when later follow-up is missing. An unmatched alert is false only with complete 45-minute follow-up, enough to resolve an onset and its 15-minute sustain requirement. Otherwise its outcome is censored and unknown, explicitly reported outside the false-warning count. Denominator remains observable non-event time under the declared rule. Unknown follow-up is not evidence of no event.

All seven saved policies were recomputed on identical saved forecast paths and thresholds. Selected boosted CGM credits 162/163 onsets (previously 161), with 268 false warnings (unchanged) and 3 censored emissions. The Type 2 subset credits 106/106, with 161 false warnings and 1 censored emission among only three participants. These are retrospective research results, not accuracy or clinical utility.

Regression tests exercise a positive onset followed by a missing boundary and a censored warning that must still suppress a later score during cooldown. The old outputs remain in reports/scoring_v1/, and every old/new aggregate is in reports/scoring_v2_audit.json. New thresholds, if investigated, must use development data and form a separate declared experiment. This correction does not reset or create an untouched test cohort.
