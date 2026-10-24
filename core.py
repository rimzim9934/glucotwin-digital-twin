"""Past-only features and incremental GlucoTwin inference. No outcomes accepted."""
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd

CLINICAL=['age','bmi','hba1c','fasting_glucose']
ACTIVITY=['hr','met','activity_calories']
HORIZONS=np.arange(5,31,5)
LAGS=[f'glucose_lag_{i*5}' for i in range(13)]
BASE=LAGS+['slope_5','slope_15','slope_30','mean_60','std_60','range_60','hour_sin','hour_cos']
DYNAMIC=[f'{c}_mean_60' for c in ACTIVITY]+[f'{c}_coverage_60' for c in ACTIVITY]

def features(records, profile):
    """13 consecutive right-edge observations, all available by issuance time."""
    if len(records)<13:return None
    r=list(records)[-13:]; times=pd.DatetimeIndex([x['time'] for x in r])
    if not np.all(np.diff(times.asi8)==300_000_000_000):return None
    g=np.asarray([x['glucose'] for x in r],float)
    if not np.isfinite(g).all():return None
    out=dict(zip(LAGS,g[::-1])); t=times[-1]; hour=t.hour+t.minute/60
    out.update(slope_5=(g[-1]-g[-2])/5,slope_15=(g[-1]-g[-4])/15,slope_30=(g[-1]-g[-7])/30,mean_60=g.mean(),std_60=g.std(),range_60=np.ptp(g),hour_sin=np.sin(hour*2*np.pi/24),hour_cos=np.cos(hour*2*np.pi/24))
    out.update({c:float(profile.get(c,np.nan)) for c in CLINICAL})
    for c in ACTIVITY:
        x=np.array([v.get(c,np.nan) for v in r],float); good=np.isfinite(x)
        out[f'{c}_mean_60']=float(x[good].mean()) if good.any() else np.nan
        out[f'{c}_coverage_60']=float(good.mean())
    return out

@dataclass
class ForecastModel:
    name: str
    columns: list
    estimator: object=None
    def predict(self,frame):
        current=frame['glucose_lag_0'].to_numpy()[:,None]
        if self.name=='persistence':return np.repeat(current,6,axis=1)
        if self.name=='trend':return current+np.clip(frame['slope_15'].to_numpy(),-2,2)[:,None]*HORIZONS
        return current+self.estimator.predict(frame[self.columns])

@dataclass
class TwinState:
    profile: dict
    model: ForecastModel
    alpha: float=0.
    interval: object=None
    history: deque=field(default_factory=lambda:deque(maxlen=13))
    pending: dict=field(default_factory=dict)
    bias: np.ndarray=field(default_factory=lambda:np.zeros(6))
    issued_count: int=0
    matured_count: int=0
    last_time: object=None
    def update(self,observation):
        """Consumes one new canonical 5-minute observation and issues a path.
        Bias uses errors of uncorrected population forecasts, exactly when observed.
        """
        obs=dict(observation); t=pd.Timestamp(obs['time']); obs['time']=t
        if self.last_time is not None and t<=self.last_time:raise ValueError('Observations must advance in time')
        self.last_time=t; actual=float(obs['glucose'])
        for target in list(self.pending):
            if target<=t:
                due=self.pending.pop(target)
                if target==t and np.isfinite(actual):
                    for h,prediction in due:
                        error=np.clip(actual-prediction,-30,30)
                        self.bias[h]=(1-self.alpha)*self.bias[h]+self.alpha*error
                        self.matured_count+=1
        self.history.append(obs); feat=features(self.history,self.profile)
        if feat is None:return {'time':t,'status':'withheld','reason':'Requires 60 minutes of uninterrupted glucose observations','forecast':None}
        population=self.model.predict(pd.DataFrame([feat]))[0]
        prediction=population+self.bias; targets=[t+pd.Timedelta(minutes=int(h)) for h in HORIZONS]
        for i,target in enumerate(targets):self.pending.setdefault(target,[]).append((i,float(population[i])))
        self.issued_count+=1
        width=np.zeros(6) if self.interval is None else np.asarray(self.interval)
        return {'time':t,'status':'supported','forecast':prediction,'population':population,'targets':targets,'lower':prediction-width,'upper':prediction+width,'bias':self.bias.copy()}

def adapt_predictions(times,glucose,population,alpha):
    """Vectorized-input reference with exactly the same matured-error rule as TwinState."""
    result=np.full_like(population,np.nan); pending={}; bias=np.zeros(6)
    for i,(t,actual) in enumerate(zip(pd.to_datetime(times),glucose)):
        for target in list(pending):
            if target<=t:
                due=pending.pop(target)
                if target==t and np.isfinite(actual):
                    for h,p in due:bias[h]=(1-alpha)*bias[h]+alpha*np.clip(actual-p,-30,30)
        if np.isfinite(population[i]).all():
            result[i]=population[i]+bias
            for h,minutes in enumerate(HORIZONS):pending.setdefault(t+pd.Timedelta(minutes=int(minutes)),[]).append((h,population[i,h]))
    return result
