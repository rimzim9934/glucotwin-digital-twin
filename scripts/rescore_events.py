"""Correct event accounting with existing saved scores and frozen thresholds.
No model fitting, forecasting changes, or threshold selection occurs here.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib,pandas as pd,numpy as np
from evaluation import score_events

def main():
    replay=joblib.load(ROOT/'artifacts/test_replay.joblib');config=json.loads((ROOT/'reports/frozen_config.json').read_text())
    path=ROOT/'reports/test_results.json';results=json.loads(path.read_text());table=pd.read_csv(ROOT/'reports/test_comparison.csv');changes=[]
    for name,p in replay['predictions'].items():
        old=results[name]['events']['overall'];new=score_events(replay['frame'],p,config['warning_thresholds'][name]);results[name]['events']=new
        for key,value in new['overall'].items():table.loc[table.model==name,key]=value
        changes.append({'model':name,'old_scoring_v1':old,'new_scoring_v2':new['overall'],'type2':new['type2']})
    path.write_text(json.dumps(results,indent=2,allow_nan=False));table.to_csv(ROOT/'reports/test_comparison.csv',index=False)
    audit={'date':'2026-10-24','scoring_version':2,'changes':'Emit warnings and advance cooldown causally. Credit observed sustained positive onsets even when later follow-up is missing. Unmatched alerts without complete 45-minute follow-up are censored.','thresholds_unchanged':config['warning_thresholds'],'models_retrained':False,'forecast_predictions_unchanged':True,'old_reports':'reports/scoring_v1/','comparisons':changes}
    (ROOT/'reports/scoring_v2_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    print(table[['model','detected_events','eligible_events','false_warnings','censored_warnings','false_warnings_per_hour']].to_string(index=False))
if __name__=='__main__':main()
