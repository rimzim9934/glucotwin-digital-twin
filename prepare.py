from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from core import CLINICAL,ACTIVITY,LAGS,BASE,DYNAMIC,HORIZONS
from evaluation import onset_indices
ROOT=Path(__file__).resolve().parent

def main():
    raw=ROOT/'data/raw/records/CGMacros'; out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    bio=pd.read_csv(raw/'bio.csv');bio.columns=bio.columns.str.strip()
    bio=bio.rename(columns={'subject':'participant','Age':'age','BMI':'bmi','A1c PDL (Lab)':'hba1c','Fasting GLU - PDL (Lab)':'fasting_glucose'})
    assert bio.participant.is_unique
    bio['cohort']=np.where(bio.hba1c>6.4,'Type 2',np.where(bio.hba1c>=5.7,'Prediabetes','Without diabetes'))
    profiles=bio[['participant','cohort']+CLINICAL].copy();profiles.to_csv(ROOT/'reports/profiles.csv',index=False)
    rng=np.random.default_rng(20261019); splits={'train':[],'validation':[],'test':[]}
    for _,group in profiles.groupby('cohort',sort=True):
        ids=rng.permutation(sorted(group.participant)).tolist();k=round(len(ids)*.2)
        splits['test']+=ids[:k];splits['validation']+=ids[k:2*k];splits['train']+=ids[2*k:]
    splits={k:sorted(v) for k,v in splits.items()};splitpath=ROOT/'reports/splits.json'
    if splitpath.exists():assert json.loads(splitpath.read_text())==splits
    else:splitpath.write_text(json.dumps(splits,indent=2))
    frames=[];audits=[]
    for profile in profiles.to_dict('records'):
        pid=profile['participant'];path=raw/f'CGMacros-{pid:03d}'/f'CGMacros-{pid:03d}.csv';x=pd.read_csv(path)
        x['time']=pd.to_datetime(x.Timestamp);duplicates=int(x.time.duplicated().sum());assert duplicates==0
        x=x.sort_values('time').set_index('time');g=pd.to_numeric(x['Dexcom GL'],errors='coerce');first=g.first_valid_index()
        if first is None:raise ValueError(f'No Dexcom for {pid}')
        initialization_end=first+pd.Timedelta(hours=24)
        init=g.loc[first:initialization_end-pd.Timedelta(minutes=1)]
        phases=(init.index.hour*60+init.index.minute)%5
        quality=[float(np.isclose(init[phases==p].dropna(),np.round(init[phases==p].dropna()),atol=1e-6).mean()) for p in range(5)]
        phase=int(np.nanargmax(quality))
        chosen=((x.index.hour*60+x.index.minute)%5)==phase
        z=pd.DataFrame({'glucose':g[chosen]});z=z.reindex(pd.date_range(z.index.min(),z.index.max(),freq='5min'))
        invalid=z.glucose.notna()&(~np.isclose(z.glucose,np.round(z.glucose),atol=1e-6)|~z.glucose.between(40,400))
        rejected=int(invalid.sum());z.loc[invalid,'glucose']=np.nan
        for original,c in [('HR','hr'),('METs','met'),('Calories (Activity)','activity_calories')]:
            series=pd.to_numeric(x.get(original,pd.Series(np.nan,index=x.index)),errors='coerce')
            if c=='met':series=series/10
            # Only minute observations at or before each grid endpoint.
            z[c]=series.rolling('5min',min_periods=1).mean().reindex(z.index)
        z.index.name='time';z=z.reset_index();z['participant']=pid;z['cohort']=profile['cohort'];z['elapsed_hours']=(z.time-first).dt.total_seconds()/3600
        for c in CLINICAL:z[c]=profile[c]
        for i,col in enumerate(LAGS):z[col]=z.glucose.shift(i)
        z['slope_5']=(z.glucose-z.glucose.shift(1))/5;z['slope_15']=(z.glucose-z.glucose.shift(3))/15;z['slope_30']=(z.glucose-z.glucose.shift(6))/30
        roll=z.glucose.rolling(13,min_periods=13);z['mean_60']=roll.mean();z['std_60']=roll.std(ddof=0);z['range_60']=roll.max()-roll.min()
        hour=z.time.dt.hour+z.time.dt.minute/60;z['hour_sin']=np.sin(hour*2*np.pi/24);z['hour_cos']=np.cos(hour*2*np.pi/24)
        for c in ACTIVITY:z[f'{c}_mean_60']=z[c].rolling(13,min_periods=1).mean();z[f'{c}_coverage_60']=z[c].notna().rolling(13,min_periods=13).mean()
        for i,h in enumerate(HORIZONS):z[f'target_{h}']=z.glucose.shift(-(i+1))
        z['supported']=z[LAGS].notna().all(axis=1)&(z.elapsed_hours>=24)
        z['split']=next(k for k,v in splits.items() if pid in v)
        audits.append(dict(participant=pid,cohort=profile['cohort'],split=z.split.iloc[0],raw_rows=len(x),duplicate_timestamps=duplicates,native_phase=phase,initial_phase_integer_fraction=quality[phase],ambiguous_samples_rejected=rejected,grid_rows=len(z),usable_glucose=int(z.glucose.notna().sum()),supported_windows=int(z.supported.sum()),monitoring_days=float((z.time.iloc[-1]-z.time.iloc[0]).total_seconds()/86400),observed_onsets=len(onset_indices(z.glucose)),glucose_min=float(z.glucose.min()),glucose_max=float(z.glucose.max())))
        frames.append(z)
    frame=pd.concat(frames,ignore_index=True);frame.to_pickle(out/'frames.pkl');pd.DataFrame(audits).to_csv(ROOT/'reports/data_audit.csv',index=False)
    (ROOT/'reports/audit_summary.json').write_text(json.dumps({'participants':len(profiles),'cohorts':profiles.cohort.value_counts().to_dict(),'raw_rows':sum(r['raw_rows'] for r in audits),'grid_rows':len(frame),'supported_windows':int(frame.supported.sum()),'observed_onsets':sum(r['observed_onsets'] for r in audits),'clinical_missingness':profiles[CLINICAL].isna().sum().to_dict(),'preprocessing':'First 24 hours establish dominant five-minute phase from integer-valued CGM endpoints. Only that phase retained thereafter; fractional or out-of-range endpoints rejected; no gap filling. Published minute stream is interpolated, so native-observation provenance cannot be fully recovered. This is reconstructed-cadence research evaluation, not raw-device validation.','hba1c_unit':'Percent, consistent with source cohort definitions and 4.6–8.5 values. Dictionary mmol/mol label is inconsistent.','source_dictionary_checksums':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw').glob('DataDictionary*.csv')}},indent=2))
    print(pd.DataFrame(audits).to_string(index=False));print('Saved',len(frame),'canonical rows',splits,flush=True)
if __name__=='__main__':main()
