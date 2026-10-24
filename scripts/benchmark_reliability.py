"""Frozen-model stress test on all reserved recordings. No tuning or training."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
from reliability import SCENARIOS,delivered_observation

def main():
    replay=joblib.load(ROOT/'artifacts/test_replay.joblib');f=replay['frame'];config=json.loads((ROOT/'reports/frozen_config.json').read_text());name=config['selected_population'];p=replay['predictions'][name][:,5]
    rows=[]
    for pid,part in f.groupby('participant',sort=False):
        ix=part.index.to_numpy();normal=part.supported.to_numpy();ready=part.elapsed_hours.to_numpy()>=24;target=part.target_30.to_numpy();original=p[ix];records=part.to_dict('records')
        for scenario in SCENARIOS:
            if scenario=='normal':good=part.glucose.notna().to_numpy();delayed=missing=0
            else:
                delivered=[delivered_observation(records,i,scenario) for i in range(len(records))]
                good=np.array([np.isfinite(x['glucose']) for x in delivered]);delayed=sum(ready[i] and x['delivery']['delivery_status']=='stale_rejected' for i,x in enumerate(delivered));missing=sum(ready[i] and x['delivery']['delivery_status']=='missing' for i,x in enumerate(delivered))
            consecutive=pd.Series(good).rolling(13,min_periods=13).sum().eq(13).to_numpy()
            supported=normal&consecutive;scored=supported&ready&np.isfinite(target)&np.isfinite(original)
            # All retained inputs are identical to baseline. Reuse the frozen path
            # exactly, rather than rerunning the same tree on identical features.
            errors=np.abs(original[scored]-target[scored])
            rows.append({'participant':int(pid),'cohort':part.iloc[0].cohort,'scenario':scenario,'monitoring_rows':int(ready.sum()),'supported_rows':int((ready&supported).sum()),'coverage':float(supported[ready].mean()) if ready.any() else None,'scored_targets':int(scored.sum()),'mae_mg_dl':float(errors.mean()) if len(errors) else None,'paired_normal_mae_mg_dl':float(errors.mean()) if len(errors) else None,'additional_withheld_rows':int((ready&normal&~supported).sum()),'stale_packets_rejected':int(delayed),'simulated_missing_packets':int(missing)})
    summary=[]
    for scenario in SCENARIOS:
        group=[r for r in rows if r['scenario']==scenario];scored=[r for r in group if r['mae_mg_dl'] is not None]
        summary.append({'scenario':scenario,'label':SCENARIOS[scenario],'reserved_patients':len(group),'scoreable_patients':len(scored),'patient_mean_coverage':float(np.mean([r['coverage'] for r in group])),'patient_mean_mae_mg_dl':float(np.mean([r['mae_mg_dl'] for r in scored])),'paired_normal_mae_mg_dl':float(np.mean([r['paired_normal_mae_mg_dl'] for r in scored])),'scored_targets':sum(r['scored_targets'] for r in group),'additional_withheld_rows':sum(r['additional_withheld_rows'] for r in group),'stale_packets_rejected':sum(r['stale_packets_rejected'] for r in group)})
    report={'date':'2026-10-24','model':name,'thresholds_retuned':False,'model_retrained':False,'scenarios':SCENARIOS,'protocol':'On the canonical 5-minute clock after 24h initialization: mask 15 minutes every 6h, or deliver 10-minute-old packets for 30 minutes every 6h. Schedule begins at phase 3h. Reject glucose older than 5 minutes. No carry-forward, timestamp relabeling, or retroactive backfill. Require 13 uninterrupted current readings after a gap. Ground-truth targets remain original observations.','interpretation':'Coverage measures all reserved participants, including one unsupported participant. MAE uses available targets for each scenario and is not comparable across different retained sets as an accuracy gain. Paired normal MAE uses identical retained targets and equals guarded MAE because retained input windows are unchanged. This is an exploratory engineering stress test on previously inspected test recordings, not an independent validation or physiological simulation.','summary':summary,'per_patient':rows}
    (ROOT/'reports/reliability_benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False));pd.DataFrame(rows).to_csv(ROOT/'reports/reliability_per_patient.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
