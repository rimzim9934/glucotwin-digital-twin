"""Offline labels and transparent alert accounting, never imported by inference."""
import numpy as np
import pandas as pd

def onset_indices(glucose):
    g=np.asarray(glucose,float); found=[]
    for i in range(1,len(g)-3):
        if np.isfinite(g[i-1]) and g[i-1]<=180 and np.all(g[i:i+4]>180):found.append(i)
    return found

def score_events(frame,predictions,threshold=180):
    """Causal emission with retrospective outcome censoring (scoring v2).
    Every qualifying supported score advances cooldown, even if its outcome is
    censored. A fully observed sustained onset can credit a warning despite a
    later gap. Only unmatched alerts need the full 45-minute outcome horizon.
    """
    records=[]; patients=[]; onset_records=[]
    for pid,part in frame.groupby('participant',sort=False):
        ix=part.index.to_numpy(); g=part.glucose.to_numpy(); p=predictions[ix]; n=len(g)
        supported=np.isfinite(p).all(axis=1); ready=part.elapsed_hours.to_numpy()>=24
        complete=np.zeros(n,bool)
        for i in range(n-9):complete[i]=np.isfinite(g[i:i+10]).all()
        opportunities=supported&ready&np.isfinite(g)&(g<=180)
        onsets=onset_indices(g); eligible=[]
        for j in onsets:
            if not ready[j]:continue
            prior=np.arange(max(0,j-6),j)
            # Onset eligibility is independent of score/threshold, but requires supported input.
            ok=prior[supported[prior]&ready[prior]&(g[prior]<=180)]
            if len(ok):eligible.append(j)
            onset_records.append({'participant':int(pid),'time':str(part.iloc[j].time),'eligible':bool(len(ok))})
        matched=set(); leads=[]; false=0; duplicate=0; censored=0; last=-999
        for i in range(n):
            if not opportunities[i] or i-last<6 or np.max(p[i])<threshold:continue
            last=i; future=[j for j in eligible if i<j<=i+6]
            if future:
                j=future[0]
                if j not in matched:matched.add(j);leads.append((j-i)*5);outcome='matched'
                else:duplicate+=1;outcome='duplicate'
            elif complete[i]:false+=1;outcome='false'
            else:censored+=1;outcome='censored'
            records.append({'participant':int(pid),'time':str(part.iloc[i].time),'current':float(g[i]),'path_max':float(np.max(p[i])),'outcome':outcome,'lead_minutes':int((future[0]-i)*5) if future else None})
        # Denominator includes unsupported but otherwise observable non-event time.
        no_event=np.array([not any(i<j<=i+6 for j in onsets) for i in range(n)])
        denom=int(np.sum(ready&complete&(g<=180)&no_event))*5/60
        patients.append({'participant':int(pid),'cohort':part.iloc[0].cohort,'eligible_events':len(eligible),'detected_events':len(matched),'missed_events':len(eligible)-len(matched),'event_sensitivity':len(matched)/len(eligible) if eligible else None,'false_warnings':false,'duplicate_warnings':duplicate,'censored_warnings':censored,'emitted_warnings':false+duplicate+censored+len(matched),'non_event_hours':denom,'false_warnings_per_hour':false/denom if denom else None,'median_lead_minutes':float(np.median(leads)) if leads else None,'coverage':float(supported[ready].mean()) if ready.any() else None,'leads':leads})
    def aggregate(rows):
        events=sum(x['eligible_events'] for x in rows);detected=sum(x['detected_events'] for x in rows);fa=sum(x['false_warnings'] for x in rows);hours=sum(x['non_event_hours'] for x in rows);leads=sum((x['leads'] for x in rows),[])
        return dict(emitted_warnings=sum(x['emitted_warnings'] for x in rows),censored_warnings=sum(x['censored_warnings'] for x in rows),duplicate_warnings=sum(x['duplicate_warnings'] for x in rows),eligible_events=events,detected_events=detected,missed_events=events-detected,event_sensitivity=detected/events if events else None,false_warnings=fa,non_event_hours=hours,false_warnings_per_hour=fa/hours if hours else None,median_lead_minutes=float(np.median(leads)) if leads else None)
    return {'scoring_version':2,'policy':'Causal emission and cooldown; matched positive onsets need observed event evidence; unmatched alerts need 45-minute complete follow-up or are censored.','overall':aggregate(patients),'type2':aggregate([r for r in patients if r['cohort']=='Type 2']),'per_patient':patients,'warnings':records,'onsets':onset_records}

def error_metrics(frame,prediction,interval=None):
    rows=[]
    for pid,part in frame.groupby('participant',sort=False):
        ix=part.index.to_numpy(); y=part.target_30.to_numpy(); p=prediction[ix,5]
        valid=np.isfinite(y)&np.isfinite(p)&(part.elapsed_hours.to_numpy()>=24)
        err=p[valid]-y[valid]
        if not len(err):
            rows.append({'participant':int(pid),'cohort':part.iloc[0].cohort,'n':0,'mae':None,'rmse':None,'coverage':0.0})
            continue
        row={'participant':int(pid),'cohort':part.iloc[0].cohort,'n':int(valid.sum()),'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2))),'coverage':float(np.isfinite(p[part.elapsed_hours.to_numpy()>=24]).mean())}
        if interval is not None:row.update(interval_coverage=float((np.abs(err)<=interval[5]).mean()),interval_width=float(2*interval[5]))
        rows.append(row)
    def aggregate(r):
        total=len(r); all_coverage=[x['coverage'] for x in r]; r=[x for x in r if x['n']>0]
        if not r:return {'patients':0,'reserved_patients':total}
        return {'reserved_patients':total,'patients':len(r),'windows':sum(x['n'] for x in r),'patient_mean_mae':float(np.mean([x['mae'] for x in r])),'patient_mean_rmse':float(np.mean([x['rmse'] for x in r])),'pooled_mae':float(np.average([x['mae'] for x in r],weights=[x['n'] for x in r])),'patient_mean_coverage':float(np.mean(all_coverage))}
    return {'overall':aggregate(rows),'type2':aggregate([r for r in rows if r['cohort']=='Type 2']),'per_patient':rows}
