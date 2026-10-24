"""Causal evidence ledger and controlled sensor-delivery experiments.

Reliability here describes observable input freshness and already matured errors,
not a probability of a correct prediction or a clinical safety judgement.
"""
from collections import deque
from copy import deepcopy
import hashlib,json
import numpy as np
import pandas as pd
from core import HORIZONS,CLINICAL

SCENARIOS={
    'normal':'Original received observations',
    'dropout':'15-minute signal loss every 6 hours',
    'delay':'10-minute delivery delay for 30 minutes every 6 hours',
}

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,deque,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(pd.Timestamp,np.datetime64)):return str(pd.Timestamp(value))
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def delivered_observation(records,index,scenario):
    """Return only data delivered by this clock tick, never future observations.

    Delay is a controlled transport backlog, not sensor physiology. Packets keep
    their original timestamps. Packets older than five minutes fail the freshness
    rule and do not replace the missing current reading or backfill the history.
    """
    if scenario not in SCENARIOS:raise ValueError('Unknown delivery scenario')
    current=deepcopy(records[index]);now=pd.Timestamp(current['time'])
    elapsed=float(current.get('elapsed_hours',0.))*60
    phase=int(round(elapsed/5))%72
    info={'clock_time':str(now),'source_time':str(now),'age_minutes':0.,'delivery_status':'current'}
    # A fixed 3-hour offset avoids masking cadence initialization. No outcomes select gaps.
    if elapsed>=24*60 and scenario=='dropout' and 36<=phase<39:
        current['glucose']=np.nan;info.update(delivery_status='missing',source_time=None,age_minutes=None)
    elif elapsed>=24*60 and scenario=='delay' and 36<=phase<42:
        old=records[index-2] if index>=2 else None
        source=pd.Timestamp(old['time']) if old is not None else None
        age=(now-source).total_seconds()/60 if source is not None else None
        current['glucose']=np.nan
        info.update(delivery_status='stale_rejected',source_time=str(source) if source is not None else None,age_minutes=age)
    current['delivery']=info
    return current


class ForecastLedger:
    def __init__(self,model_name,profile,model_fingerprint=None):
        self.model_name=model_name;self.profile=clean(profile);self.model_fingerprint=model_fingerprint
        self.history=deque(maxlen=13);self.receipts=[];self.pending={};self.recent=deque(maxlen=20)
        self.last_delivery=None

    def observe(self,observation,output):
        obs=clean(observation);now=pd.Timestamp(obs['time']);actual=obs.get('glucose')
        self.last_delivery=obs.get('delivery',{'clock_time':str(now),'source_time':str(now),'age_minutes':0.,'delivery_status':'current' if actual is not None else 'missing'})
        # Only an actually received target reading can mature a receipt's outcome.
        for target in list(self.pending):
            if target>now:continue
            refs=self.pending.pop(target)
            for receipt,h in refs:
                issued=receipt['issued_forecast'][h]
                outcome={'target_time':str(target),'received_at':str(now),'observed_glucose':actual if target==now else None,'status':'observed' if target==now and actual is not None else 'unobserved'}
                outcome['absolute_error']=abs(issued['glucose_mg_dl']-actual) if outcome['status']=='observed' else None
                receipt['outcomes'][h]=outcome
                if h==5 and outcome['absolute_error'] is not None:self.recent.append({'origin_time':receipt['issued_at'],'target_time':str(target),'absolute_error':outcome['absolute_error']})
        self.history.append({k:obs.get(k) for k in ['time','glucose','hr','met','activity_calories','delivery']})
        if output['forecast'] is None:return
        path=[{'target_time':str(t),'glucose_mg_dl':float(v)} for t,v in zip(output['targets'],output['forecast'])]
        payload={'schema_version':1,'issued_at':str(now),'model_name':self.model_name,'model_artifact_sha256':self.model_fingerprint,'clinical_profile':self.profile,'clinical_profile_used':self.model_name in {'clinical_cgm','clinical_activity'},'received_input_window':clean(list(self.history)),'delivery_at_issue':deepcopy(self.last_delivery),'issued_forecast':path,'recent_performance_at_issue':self.summary(),'scope':'Recorded research observations. Forecast and input snapshot fixed at issuance. Outcomes append only after target readings arrive. No treatment advice.'}
        digest=hashlib.sha256(json.dumps(clean(payload),sort_keys=True,allow_nan=False).encode()).hexdigest()
        receipt={'receipt_id':digest[:16],'issued_payload_sha256':digest,**payload,'outcomes':[None]*6}
        self.receipts.append(receipt)
        for h,item in enumerate(path):self.pending.setdefault(pd.Timestamp(item['target_time']),[]).append((receipt,h))

    def summary(self):
        values=[x['absolute_error'] for x in self.recent]
        return {'observed_matured_30min_forecasts':len(values),'recent_30min_mae_mg_dl':float(np.mean(values)) if values else None,'window':'Most recent 20 observed matured 30-minute forecasts. Overlapping and descriptive, not calibration or a safety score.'}

    def export(self):
        return json.dumps(clean({'schema_version':1,'model':self.model_name,'recent_performance':self.summary(),'receipts':self.receipts}),indent=2,allow_nan=False)


def simulate_segment(records,profile,model,scenario,interval=None,fingerprint=None):
    from core import TwinState
    state=TwinState(profile,model,alpha=0,interval=interval)
    ledger=ForecastLedger(model.name,{k:profile.get(k) for k in CLINICAL},fingerprint)
    rows=[]
    for i in range(len(records)):
        obs=delivered_observation(records,i,scenario);out=state.update(obs);ledger.observe(obs,out)
        if float(obs.get('elapsed_hours',24))<24:continue
        rows.append({'time':str(obs['time']),'received_glucose':clean(obs.get('glucose')),'forecast_30':float(out['forecast'][-1]) if out['forecast'] is not None else None,'status':out['status'],'reason':out.get('reason','Complete fresh glucose history'),'delivery_status':obs['delivery']['delivery_status'],'source_time':obs['delivery']['source_time'],'age_minutes':obs['delivery']['age_minutes'],'recent_mae':ledger.summary()['recent_30min_mae_mg_dl']})
    return pd.DataFrame(rows),ledger
