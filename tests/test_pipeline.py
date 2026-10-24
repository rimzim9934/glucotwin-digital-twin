import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from core import TwinState,ForecastModel,features,adapt_predictions,BASE
from evaluation import onset_indices,score_events

def observations(n=30):
    return [{'time':t,'glucose':100.+i,'hr':70.,'met':1.,'activity_calories':1.} for i,t in enumerate(pd.date_range('2026-01-01',periods=n,freq='5min'))]

def test_future_changes_do_not_change_issued_forecasts():
    a=observations();b=observations();b[25]['glucose']=350
    def run(r):
        s=TwinState({},ForecastModel('trend',BASE),alpha=.1)
        return [s.update(x) for x in r]
    x,y=run(a),run(b)
    for i in range(25):
        if x[i]['forecast'] is not None:np.testing.assert_allclose(x[i]['forecast'],y[i]['forecast'])

def test_gap_withholds_then_requires_new_history():
    r=observations(40);r[20]['glucose']=np.nan;s=TwinState({},ForecastModel('persistence',BASE));out=[s.update(x) for x in r]
    assert all(v['status']=='withheld' for v in out[20:33]);assert out[33]['status']=='supported'

def test_only_matured_errors_change_bias_and_chunk_parity():
    r=observations();s=TwinState({},ForecastModel('persistence',BASE),alpha=.1);out=[s.update(x) for x in r]
    assert np.all(out[12]['bias']==0);assert out[13]['bias'][0]!=0;assert out[13]['bias'][5]==0
    population=np.full((len(r),6),np.nan)
    for i,x in enumerate(out):
        if x['forecast'] is not None:population[i]=x['population']
    p=adapt_predictions([x['time']for x in r],[x['glucose']for x in r],population,.1)
    for i,x in enumerate(out):
        if x['forecast'] is not None:np.testing.assert_allclose(x['forecast'],p[i])
    chunked=TwinState({},ForecastModel('persistence',BASE),alpha=.1);q=[]
    for chunk in [r[:7],r[7:19],r[19:]]:q.extend(chunked.update(x)for x in chunk)
    np.testing.assert_allclose(q[-1]['forecast'],out[-1]['forecast'])

def test_event_requires_new_crossing_and_sustained_observations():
    assert onset_indices([190]*5+[170,181,190,195,185])==[6]
    assert onset_indices([170,181,190,np.nan,195])==[]

def test_alert_matching_and_cooldown():
    g=np.array([150.]*10+[190.]*4+[150.]*20);f=pd.DataFrame({'participant':1,'cohort':'Type 2','time':pd.date_range('2026-01-01',periods=len(g),freq='5min'),'glucose':g,'elapsed_hours':25.})
    p=np.full((len(g),6),100.);p[5:10]=190;p[20]=190
    r=score_events(f,p,180)
    assert r['overall']['detected_events']==1;assert r['overall']['false_warnings']==1
    assert len(r['warnings'])==2;assert r['overall']['median_lead_minutes']==25

def test_actual_feature_and_split_parity():
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/processed/frames.pkl').exists():return
    import json
    splits=json.loads((root/'reports/splits.json').read_text());a,b,c=map(set,[splits['train'],splits['validation'],splits['test']]);assert not(a&b or a&c or b&c)
    frame=pd.read_pickle(root/'data/processed/frames.pkl')
    from core import CLINICAL,DYNAMIC
    for _,part in frame.groupby('participant'):
        part=part.reset_index(drop=True)
        if not part.supported.any():continue
        idx=part.index[part.supported][0];window=part.iloc[idx-12:idx+1];computed=features(window.to_dict('records'),part.iloc[idx].to_dict())
        for col in BASE+CLINICAL+DYNAMIC:np.testing.assert_allclose(computed[col],part.iloc[idx][col],rtol=1e-8,atol=1e-8,equal_nan=True)

def test_trained_model_incremental_matches_offline():
    import joblib
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/models.joblib').exists():return
    models=joblib.load(root/'artifacts/models.joblib');f=pd.read_pickle(root/'data/processed/frames.pkl');part=f[(f.split=='validation')].groupby('participant').get_group(int(f[f.split=='validation'].participant.iloc[0])).reset_index(drop=True)
    first=int(part.index[part.supported][0]);records=part.iloc[first-12:first+40].to_dict('records')
    for name in ['boosted_cgm','clinical_activity']:
        state=TwinState(part.iloc[first].to_dict(),models[name]);outputs=[state.update(x) for x in records]
        for offset,out in enumerate(outputs[12:]):
            row=part.iloc[[first+offset]]
            if out['forecast'] is not None:np.testing.assert_allclose(out['forecast'],models[name].predict(row)[0],atol=1e-8)

def test_positive_warning_survives_later_missing_followup():
    g=np.full(30,150.);g[10:14]=190.;g[15]=np.nan
    f=pd.DataFrame({'participant':1,'cohort':'Type 2','time':pd.date_range('2026-01-01',periods=len(g),freq='5min'),'glucose':g,'elapsed_hours':25.})
    p=np.full((len(g),6),100.);p[8]=190.
    r=score_events(f,p,180)
    assert r['overall']['detected_events']==1
    assert r['warnings'][0]['outcome']=='matched'
    assert r['warnings'][0]['lead_minutes']==10


def test_censored_warning_advances_causal_cooldown():
    g=np.full(35,150.);g[8]=np.nan
    f=pd.DataFrame({'participant':1,'cohort':'Type 2','time':pd.date_range('2026-01-01',periods=len(g),freq='5min'),'glucose':g,'elapsed_hours':25.})
    p=np.full((len(g),6),100.);p[[4,9,10]]=190.
    r=score_events(f,p,180)
    assert [x['time'] for x in r['warnings']]==[str(f.iloc[4].time),str(f.iloc[10].time)]
    assert [x['outcome'] for x in r['warnings']]==['censored','false']
    assert r['overall']['censored_warnings']==1 and r['overall']['false_warnings']==1
