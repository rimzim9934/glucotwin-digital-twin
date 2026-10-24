"""Run development first, freeze configuration, then evaluate test exactly once."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import argparse,json,hashlib
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from core import ForecastModel,BASE,CLINICAL,DYNAMIC,HORIZONS,adapt_predictions
from evaluation import error_metrics,score_events
ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts';REP=ROOT/'reports';ART.mkdir(exist_ok=True)
def dump(name,value): (REP/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def predict(model,f):
    p=np.full((len(f),6),np.nan);ok=f.supported.to_numpy();p[ok]=model.predict(f.loc[ok]);return p

def adapt(f,p,alpha):
    output=np.full_like(p,np.nan)
    for _,part in f.groupby('participant',sort=False):
        ix=part.index.to_numpy();output[ix]=adapt_predictions(part.time,part.glucose.to_numpy(),p[ix],alpha)
    return output

def dev():
    f=pd.read_pickle(ROOT/'data/processed/frames.pkl');tr=f[f.split=='train'].reset_index(drop=True);va=f[f.split=='validation'].reset_index(drop=True)
    target=[f'target_{h}' for h in HORIZONS];valid=tr.supported&tr[target].notna().all(axis=1);x=tr[valid];y=x[target].to_numpy()-x.glucose.to_numpy()[:,None]
    counts=x.participant.value_counts();w=x.participant.map(lambda i:len(x)/(len(counts)*counts[i])).to_numpy()
    models={'persistence':ForecastModel('persistence',BASE),'trend':ForecastModel('trend',BASE)}
    specs={'ridge_cgm':BASE,'boosted_cgm':BASE,'clinical_cgm':BASE+CLINICAL,'clinical_activity':BASE+CLINICAL+DYNAMIC}
    for name,cols in specs.items():
        print('Training',name,len(x),'windows',flush=True)
        if name=='ridge_cgm':
            estimator=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.))
            estimator.fit(x[cols],y,ridge__sample_weight=w)
        else:
            estimator=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=80,l2_regularization=10,learning_rate=.06,early_stopping=False,random_state=20261019))
            estimator.fit(x[cols],y,sample_weight=w)
        models[name]=ForecastModel(name,cols,estimator)
    rows=[];preds={}
    for name,model in models.items():
        p=predict(model,va);preds[name]=p;m=error_metrics(va,p);rows.append({'model':name,**m['overall'],'type2_mae':m['type2'].get('patient_mean_mae')});print(name,m['overall'],flush=True)
    best=min(rows,key=lambda r:r['patient_mean_mae'])['model'];alpha_results=[]
    for alpha in [0.,.03,.1]:
        p=adapt(va,preds[best],alpha);m=error_metrics(va,p);alpha_results.append({'alpha':alpha,'mae':m['overall']['patient_mean_mae']})
    alpha=min(alpha_results,key=lambda r:r['mae'])['alpha'];chosen=adapt(va,preds[best],alpha)
    widths=[]
    for i,h in enumerate(HORIZONS):
        e=np.abs(chosen[:,i]-va[f'target_{h}'].to_numpy());e=e[np.isfinite(e)];widths.append(float(np.quantile(e,.9,method='higher')))
    policies={};eventrows=[]
    for name,p in {**preds,'adaptive_twin':chosen}.items():
        options=[]
        for threshold in [170,180,190,200]:
            m=score_events(va,p,threshold)['overall'];options.append({'threshold':threshold,**m})
        feasible=[r for r in options if r['false_warnings_per_hour'] is not None and r['false_warnings_per_hour']<=.25]
        if feasible:policy=max(feasible,key=lambda r:(r['event_sensitivity'] or 0,-r['false_warnings_per_hour']))
        else:policy=min(options,key=lambda r:(r['false_warnings_per_hour'] if r['false_warnings_per_hour'] is not None else 1e9,-(r['event_sensitivity'] or 0)))
        policies[name]=policy['threshold'];eventrows.extend([{'model':name,**r} for r in options])
    config={'selected_population':best,'adaptation_alpha':alpha,'interval_half_widths':widths,'warning_thresholds':policies,'selection':'Equal-patient validation MAE at 30 minutes; no test model selection','validation_baselines':rows,'adaptation_candidates':alpha_results,'protocol_sha256':hashlib.sha256((ROOT/'docs/PROTOCOL.md').read_bytes()).hexdigest(),'splits_sha256':hashlib.sha256((REP/'splits.json').read_bytes()).hexdigest()}
    dump('frozen_config.json',config);pd.DataFrame(rows).to_csv(REP/'validation_forecasts.csv',index=False);pd.DataFrame(eventrows).to_csv(REP/'validation_warning_policies.csv',index=False)
    joblib.dump(models,ART/'models.joblib');joblib.dump({'model':models[best],'alpha':alpha,'interval':np.array(widths),'threshold':policies['adaptive_twin']},ART/'deployment.joblib');print('FROZEN',json.dumps(config),flush=True)

def final(verify=False):
    if (REP/'test_results.json').exists() and not verify:raise RuntimeError('Final test already evaluated. Preserve results; use a new declared experiment for further work.')
    config=json.loads((REP/'frozen_config.json').read_text());assert hashlib.sha256((ROOT/'docs/PROTOCOL.md').read_bytes()).hexdigest()==config['protocol_sha256']
    f=pd.read_pickle(ROOT/'data/processed/frames.pkl');te=f[f.split=='test'].reset_index(drop=True);models=joblib.load(ART/'models.joblib');preds={name:predict(m,te) for name,m in models.items()};preds['adaptive_twin']=adapt(te,preds[config['selected_population']],config['adaptation_alpha'])
    results={};rows=[];patients=[]
    for name,p in preds.items():
        errors=error_metrics(te,p,config['interval_half_widths'] if name=='adaptive_twin' else None);events=score_events(te,p,config['warning_thresholds'][name]);results[name]={'forecast':errors,'events':events}
        rows.append({'model':name,**errors['overall'],'type2_mae':errors['type2'].get('patient_mean_mae'),**events['overall'],'threshold':config['warning_thresholds'][name]})
        patients.extend([{'model':name,**r} for r in errors['per_patient']]);print(name,rows[-1],flush=True)
    if verify:
        existing=json.loads((REP/'test_results.json').read_text())
        for name in results:
            np.testing.assert_allclose(results[name]['forecast']['overall']['patient_mean_mae'],existing[name]['forecast']['overall']['patient_mean_mae'],rtol=1e-8)
            assert results[name]['events']['overall']==existing[name]['events']['overall']
        print('REPRODUCTION MATCHES RECORDED RESULTS',flush=True)
    else:dump('test_results.json',results)
    pd.DataFrame(rows).to_csv(REP/'test_comparison.csv',index=False);pd.DataFrame(patients).to_csv(REP/'test_per_patient.csv',index=False)
    joblib.dump({'frame':te,'predictions':preds},ART/'test_replay.joblib');print('FINAL TEST COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dev','final']);p.add_argument('--verify',action='store_true');args=p.parse_args();dev() if args.stage=='dev' else final(args.verify)
