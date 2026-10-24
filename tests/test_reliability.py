import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from core import TwinState,ForecastModel,BASE
from reliability import ForecastLedger,delivered_observation,simulate_segment,clean

def records(n=110):
    return [{'time':t,'glucose':100.+i/5,'elapsed_hours':24+i/12,'hr':70.,'met':1.,'activity_calories':1.} for i,t in enumerate(pd.date_range('2026-01-01',periods=n,freq='5min'))]

def test_receipt_input_snapshot_is_fixed_and_outcomes_mature_only_on_arrival():
    r=records(25);state=TwinState({},ForecastModel('persistence',BASE));ledger=ForecastLedger('persistence',{})
    for obs in r[:13]:ledger.observe(obs,state.update(obs))
    receipt=ledger.receipts[0];issued=json.dumps(receipt['issued_forecast']);inputs=json.dumps(receipt['received_input_window'])
    assert ledger.summary()['observed_matured_30min_forecasts']==0
    for obs in r[13:18]:ledger.observe(obs,state.update(obs))
    assert receipt['outcomes'][5] is None
    assert ledger.summary()['observed_matured_30min_forecasts']==0
    ledger.observe(r[18],state.update(r[18]))
    assert receipt['outcomes'][5]['status']=='observed'
    assert ledger.summary()['observed_matured_30min_forecasts']==1
    assert json.dumps(receipt['issued_forecast'])==issued and json.dumps(receipt['received_input_window'])==inputs
    payload={k:v for k,v in receipt.items() if k not in {'receipt_id','issued_payload_sha256','outcomes'}}
    assert hashlib.sha256(json.dumps(clean(payload),sort_keys=True,allow_nan=False).encode()).hexdigest()==receipt['issued_payload_sha256']
    assert max(pd.Timestamp(x['time']) for x in receipt['received_input_window'])<=pd.Timestamp(receipt['issued_at'])
    json.loads(ledger.export())


def test_missing_target_does_not_count_as_zero_recent_error():
    r=records(20);r[18]['glucose']=np.nan;state=TwinState({},ForecastModel('persistence',BASE));ledger=ForecastLedger('persistence',{})
    for obs in r[:19]:ledger.observe(obs,state.update(obs))
    assert ledger.receipts[0]['outcomes'][5]['status']=='unobserved'
    assert ledger.summary()['observed_matured_30min_forecasts']==0


def test_delivery_delays_keep_source_time_and_require_fresh_history_for_recovery():
    r=records();late=delivered_observation(r,36,'delay')
    assert late['delivery']['age_minutes']==10 and late['delivery']['source_time']==str(r[34]['time'])
    assert np.isnan(late['glucose']) and late['time']==r[36]['time']
    rows,_=simulate_segment(r,{},ForecastModel('persistence',BASE),'delay')
    assert rows.iloc[35]['status']=='supported'
    assert (rows.iloc[36:54].status=='withheld').all()
    assert rows.iloc[54]['status']=='supported'
    lost,_=simulate_segment(r,{},ForecastModel('persistence',BASE),'dropout')
    assert (lost.iloc[36:51].status=='withheld').all() and lost.iloc[51]['status']=='supported'


def test_normal_retained_forecasts_match_both_disrupted_runs_and_future_isolation():
    r=records();model=ForecastModel('trend',BASE)
    normal,_=simulate_segment(r,{},model,'normal')
    for scenario in ['dropout','delay']:
        changed,_=simulate_segment(r,{},model,scenario)
        ok=changed.forecast_30.notna()
        np.testing.assert_allclose(changed.loc[ok,'forecast_30'],normal.loc[ok,'forecast_30'])
        altered=records();altered[80]['glucose']=350.
        future,_=simulate_segment(altered,{},model,scenario)
        np.testing.assert_allclose(changed.iloc[:80].forecast_30,future.iloc[:80].forecast_30,equal_nan=True)


def test_full_benchmark_mask_matches_incremental_trained_model():
    import joblib
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/test_replay.joblib').exists():return
    replay=joblib.load(root/'artifacts/test_replay.joblib');models=joblib.load(root/'artifacts/models.joblib')
    frame=replay['frame'];part=frame[frame.participant==42].reset_index(drop=True)
    first=int(part.index[part.supported][0]);part=part.iloc[first-12:first+90].reset_index(drop=True);records=part.to_dict('records')
    expected=models['boosted_cgm'].predict(part)
    for scenario in ['normal','dropout','delay']:
        rows,_=simulate_segment(records,{},models['boosted_cgm'],scenario)
        received=[delivered_observation(records,i,scenario) for i in range(len(records))]
        good=pd.Series([np.isfinite(x['glucose']) for x in received]).rolling(13,min_periods=13).sum().eq(13).to_numpy()
        ready=part.elapsed_hours.to_numpy()>=24
        assert np.array_equal(rows.forecast_30.notna().to_numpy(),good[ready])
        np.testing.assert_allclose(rows.loc[good[ready],'forecast_30'],expected[ready][good[ready],5],atol=1e-8)
