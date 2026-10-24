"""GlucoTwin research dashboard — actual incremental inference over recorded data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import json,hashlib
from reliability import ForecastLedger,SCENARIOS,simulate_segment
import joblib,numpy as np,pandas as pd,plotly.graph_objects as go,streamlit as st
from core import TwinState,CLINICAL
st.set_page_config(page_title='GlucoTwin · Patient Intelligence',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}.stApp{background:#0b1220;color:#e5edf7}h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.04em}h1{font-size:2.8rem!important;font-weight:800!important}section[data-testid="stSidebar"]{background:#101b2d;border-right:1px solid #203149}div[data-testid="stMetric"]{background:#121f31;border:1px solid #26384e;border-radius:14px;padding:18px}div[data-testid="stMetricLabel"]{color:#a6b6cb}div[data-testid="stMetricValue"]{font-family:'Manrope';font-size:2rem}div[data-testid="stMetricDelta"]{font-size:.8rem}.block-container{padding-top:4.5rem;max-width:1500px}.eyebrow{font-size:11px;letter-spacing:.19em;color:#51d6c7;font-weight:700}.subtitle{color:#9eafc5;font-size:16px;max-width:850px}.panel{border:1px solid #26384e;background:#121f31;padding:20px;border-radius:14px}.tag{background:#173b3b;color:#71e4d5;padding:5px 10px;border-radius:20px;font-size:11px;letter-spacing:.06em}.note{font-size:12px;color:#8fa4bd}.stButton>button{border-radius:9px}.stTabs [data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #26384e}.stTabs [data-baseweb="tab"]{font-size:14px}div[data-testid="stAlert"] p{color:#e5edf7}footer{visibility:hidden}
</style>''',unsafe_allow_html=True)
@st.cache_resource
def load():
    replay=joblib.load(ROOT/'artifacts/test_replay.joblib');models=joblib.load(ROOT/'artifacts/models.joblib');config=json.loads((ROOT/'reports/frozen_config.json').read_text());results=json.loads((ROOT/'reports/test_results.json').read_text());return replay,models,config,results
if not (ROOT/'artifacts/test_replay.joblib').exists():
    st.title('GlucoTwin');st.info('The evaluation pipeline is running. This screen will be ready when final results are saved.');st.stop()
replay,models,config,results=load();model_fingerprint=hashlib.sha256((ROOT/'artifacts/models.joblib').read_bytes()).hexdigest();frame=replay['frame'];names={'boosted_cgm':'CGM forecast · validation winner','clinical_cgm':'Clinical + CGM','clinical_activity':'Clinical + CGM + activity','ridge_cgm':'Linear CGM model','persistence':'Unchanged glucose baseline','trend':'Recent trend baseline'}
with st.sidebar:
    st.markdown('### ◈ GlucoTwin');st.caption('PATIENT INTELLIGENCE / RESEARCH')
    st.divider();st.markdown('**Patient workspace**')
    ids=sorted(frame.participant.unique());default=next((i for i,p in enumerate(ids) if frame[frame.participant==p].cohort.iloc[0]=='Type 2'),0)
    pid=st.selectbox('Anonymous participant',ids,index=default,format_func=lambda p:f'P{p:03d} · {frame[frame.participant==p].cohort.iloc[0]}')
    model_name=st.selectbox('Forecast engine',list(names),format_func=lambda x:names[x])
    st.caption('Clinical fusion is available for comparison. Its added benefit was not established on validation.')
    st.divider();st.markdown('**Replay controls**')
    scenario=st.selectbox('Review scenario',['Start of supported monitoring','Example: successful warning','Example: missed episode','Example: false warning'])
    reveal=st.toggle('Reveal future outcomes',value=False)
    loss=st.toggle('Simulate CGM signal loss',value=False)
    st.caption('Research recordings, not a live connected sensor. Scenario selection uses outcome annotations; inference receives past readings only.')
    st.divider();st.markdown('<span class="tag">CGMACROS v1.0.0</span>',unsafe_allow_html=True);st.caption('45 participants · 14 Type 2\n\nUS research cohort. No Indian-population validation. No dosing recommendations.')
part=frame[frame.participant==pid].reset_index(drop=True);profile=part.iloc[0].to_dict();supported=np.flatnonzero(part.supported)
key=(int(pid),model_name,scenario)
if st.session_state.get('replay_key')!=key:
    first=int(supported[0]) if len(supported) else min(300,len(part)-1)
    eventinfo=results[model_name]['events'];case_time=None
    if scenario=='Example: successful warning':
        choices=[w for w in eventinfo['warnings'] if w['participant']==pid and w['outcome']=='matched'];case_time=choices[0]['time'] if choices else None
    elif scenario=='Example: false warning':
        choices=[w for w in eventinfo['warnings'] if w['participant']==pid and w['outcome']=='false'];case_time=choices[0]['time'] if choices else None
    elif scenario=='Example: missed episode':
        warnings=[w for w in eventinfo['warnings'] if w['participant']==pid and w['outcome']=='matched']
        detected={pd.Timestamp(w['time'])+pd.Timedelta(minutes=w['lead_minutes']) for w in warnings}
        choices=[e for e in eventinfo['onsets'] if e['participant']==pid and e['eligible'] and pd.Timestamp(e['time']) not in detected];case_time=str(pd.Timestamp(choices[0]['time'])-pd.Timedelta(minutes=20)) if choices else None
    if case_time is not None:first=max(first,int(np.searchsorted(part.time.to_numpy(),pd.Timestamp(case_time).to_datetime64())))
    state=TwinState({c:profile[c] for c in CLINICAL},models[model_name],alpha=0,interval=config['interval_half_widths'] if model_name==config['selected_population'] else None)
    ledger=ForecastLedger(model_name,{c:profile[c] for c in CLINICAL},model_fingerprint)
    for record in part.iloc[max(0,first-12):first+1].to_dict('records'):
        out=state.update(record);ledger.observe(record,out)
    st.session_state.update(replay_key=key,twin=state,ledger=ledger,cursor=first,current=out,playing=False,issued=[out] if out['forecast'] is not None else [],last_alert=None,alerts=[],lost_indices=[],scenario_missing=scenario!='Start of supported monitoring' and case_time is None)

def step(count=1):
    for _ in range(count):
        i=st.session_state.cursor+1
        if i>=len(part):st.session_state.playing=False;break
        obs=part.iloc[i].to_dict()
        if loss:
            obs['glucose']=np.nan
            st.session_state.lost_indices.append(i)
        out=st.session_state.twin.update(obs);st.session_state.ledger.observe(obs,out);st.session_state.cursor=i;st.session_state.current=out
        if out['forecast'] is not None:
            st.session_state.issued.append(out)
            last=st.session_state.last_alert;threshold=config['warning_thresholds'][model_name]
            if obs['glucose']<=180 and np.max(out['forecast'])>=threshold and (last is None or (out['time']-last).total_seconds()>=1800):
                st.session_state.alerts.append(out['time']);st.session_state.last_alert=out['time']

st.markdown('<div class="eyebrow">GLUCOTWIN / RESEARCH WORKSPACE</div>',unsafe_allow_html=True)
st.title('Forecast the next 30 minutes.')
st.markdown('<div class="subtitle">An evolving glucose model, a 30-minute forecast, and an honest view of the evidence behind every warning.</div>',unsafe_allow_html=True)
st.write('')
tab1,tab2,tab3,tab4=st.tabs(['Patient twin','Performance & evidence','Data & methodology','Reliability lab'])
with tab1:
    a,b=st.columns([3,1]);a.subheader(f'P{pid:03d} / {profile["cohort"]}');b.caption('REPLAY • 5-MINUTE STEPS')
    c=st.columns(4)
    for col,label,value in zip(c,['Age','BMI','HbA1c','Fasting glucose'],[f'{profile["age"]:.0f} years',f'{profile["bmi"]:.1f}',f'{profile["hba1c"]:.1f}%',f'{profile["fasting_glucose"]:.0f} mg/dL']):col.caption(label);col.markdown(f'**{value}**')
    st.write('')
    if st.session_state.scenario_missing:st.info('This participant has no example of that outcome under the selected policy. Showing the start of monitoring.')
    controls=st.columns([1,1,1,1,3])
    def toggle_play():
        st.session_state.playing=not st.session_state.playing
    controls[0].button('Pause' if st.session_state.playing else '▶ Play',width='stretch',on_click=toggle_play)
    if controls[1].button('+5 min',width='stretch'):step()
    if controls[2].button('+30 min',width='stretch'):step(6)
    if controls[3].button('↺ Reset',width='stretch'):st.session_state.replay_key=None;st.rerun()
    controls[4].caption('Each step feeds a new observation into the inference function. Bias correction is disabled: validation did not support it.')
    @st.fragment(run_every=.8 if st.session_state.playing else None)
    def monitor():
        if st.session_state.playing:step()
        i=st.session_state.cursor;out=st.session_state.current;t=part.iloc[i].time;current=part.iloc[i].glucose if i not in st.session_state.lost_indices else np.nan
        st.caption(f'Recording time: {t:%d %b %Y · %H:%M} · shifted research dates · {names[model_name]}')
        supported_now=out['forecast'] is not None
        cols=st.columns(4);cols[0].metric('Latest received glucose',f'{current:.0f} mg/dL' if np.isfinite(current) else 'Unavailable')
        cols[1].metric('Forecast · +30 minutes',f'{out["forecast"][-1]:.0f} mg/dL' if supported_now else 'Withheld')
        cols[2].metric('Patient state','Updating' if supported_now else 'Insufficient data')
        cols[3].metric('Forecasts issued',str(st.session_state.twin.issued_count))
        if loss:st.warning('Signal-loss simulation is active for incoming readings. Turn it off, then accumulate 60 minutes of valid history to resume forecasts.')
        if not supported_now:st.info(out['reason'])
        elif current>180:st.info('Currently above the research event threshold. This is an ongoing episode, not a new-onset warning opportunity.')
        elif max(out['forecast'])>=config['warning_thresholds'][model_name]:st.warning(f'Advance-warning criterion met: forecast path reaches {max(out["forecast"]):.0f} mg/dL. The validation-selected alert threshold is {config["warning_thresholds"][model_name]} mg/dL; the research event threshold is >180 mg/dL. Repeat alerts are limited to one per 30 minutes.')
        else:st.success('Forecast is below the warning criterion. This is a model output, not an assurance of safety.')
        visible=part.iloc[max(0,i-36):i+1].copy();visible.loc[visible.index.isin(st.session_state.lost_indices),'glucose']=np.nan;fig=go.Figure()
        fig.add_trace(go.Scatter(x=visible.time,y=visible.glucose,name='Observed CGM',mode='lines',line=dict(color='#67e2d0',width=3)))
        for previous in st.session_state.issued[-12:-1]:
            fig.add_trace(go.Scatter(x=previous['targets'],y=previous['forecast'],mode='lines',line=dict(color='rgba(122,161,231,.18)',width=1),showlegend=False,hoverinfo='skip'))
        if supported_now:
            if model_name==config['selected_population']:
                fig.add_trace(go.Scatter(x=out['targets'],y=out['upper'],mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=out['targets'],y=out['lower'],mode='lines',fill='tonexty',fillcolor='rgba(134,167,255,.16)',line=dict(width=0),name='Validation residual band · nominal 90%'))
            fig.add_trace(go.Scatter(x=[t]+out['targets'],y=[current]+list(out['forecast']),name='Issued forecast',mode='lines+markers',line=dict(color='#9bb5ff',width=3,dash='dot'),marker=dict(size=5)))
        if reveal:
            future=part.iloc[i+1:i+7];fig.add_trace(go.Scatter(x=future.time,y=future.glucose,name='Future outcome · revealed',mode='lines+markers',line=dict(color='#f3bf6a',width=2)))
        fig.add_hline(y=180,line_dash='dash',line_color='#b78554',annotation_text='Research event threshold · 180')
        fig.add_vline(x=t.timestamp()*1000,line_color='#5b6d84',line_dash='dot')
        fig.update_layout(height=430,paper_bgcolor='#0b1220',plot_bgcolor='#0b1220',font=dict(color='#aebed3',family='DM Sans'),margin=dict(l=15,r=20,t=25,b=10),legend=dict(orientation='h',y=1.13),hovermode='x unified',xaxis=dict(range=[t-pd.Timedelta(hours=3),t+pd.Timedelta(minutes=35)],gridcolor='#1b2b40',title='Shifted recording time'),yaxis=dict(title='Glucose · mg/dL',gridcolor='#1b2b40'))
        st.plotly_chart(fig,width='stretch',key='forecast_chart')
        st.caption('Faint lines are previously issued forecasts. Future outcomes are hidden unless revealed. '+('The band is an empirical validation-error interval; test coverage is reported in Evidence.' if model_name==config['selected_population'] else 'No uncertainty band was calibrated for this comparison engine.'))
        evidence=st.session_state.ledger.summary()
        st.markdown('**Recent observed performance**')
        if evidence['recent_30min_mae_mg_dl'] is None:
            st.caption('Waiting for a 30-minute target reading to arrive. No observed error is available yet.')
        else:
            st.write(f"Recent 30-minute MAE: **{evidence['recent_30min_mae_mg_dl']:.1f} mg/dL** over **{evidence['observed_matured_30min_forecasts']}** observed matured forecasts.")
        st.caption('Most recent 20 observed matured forecasts in this replay only. Overlapping outcomes, descriptive error, not a confidence probability or safety score. Missing targets do not count as zero error.')
        st.download_button('Download replay forecast receipts (JSON)',st.session_state.ledger.export(),f'P{pid:03d}_{model_name}_forecast_receipts.json','application/json',key='replay_receipts')
        left,right=st.columns(2)
        with left:
            st.markdown('**What the twin knows now**');recent=part.iloc[max(0,i-12):i+1]
            st.write(f'Clinical profile linked to P{pid:03d}. Recent glucose, trend and variability update with every reading.')
            st.caption(f'Available heart-rate values in recent window: {recent.hr.notna().sum()}/{len(recent)}. Clinical/activity fields affect the corresponding fusion engines only.')
        with right:
            st.markdown('**Observe → forecast → compare**')
            matured=[x for x in st.session_state.issued if x['targets'][-1]<=t]
            if matured:
                last=matured[-1];actual=part.loc[part.time==last['targets'][-1],'glucose']
                if len(actual) and np.isfinite(actual.iloc[0]):st.write(f'Last matured 30-minute forecast: {last["forecast"][-1]:.1f}. Recorded: {actual.iloc[0]:.1f} mg/dL. Absolute error: {abs(last["forecast"][-1]-actual.iloc[0]):.1f}.')
                else:st.write('No observed target reading for the last matured forecast.')
            else:st.write('Advance the replay by 30 minutes to inspect the first matured forecast.')
    monitor()
with tab2:
    st.subheader('Evidence before claims')
    st.caption('Models and warning thresholds were selected on 9 validation patients, then evaluated once on 9 reserved patients. One reserved patient has no supported windows under the fixed cadence rule, leaving 8 with forecast-error measurements. All 9 remain in monitoring coverage.')
    table=pd.read_csv(ROOT/'reports/test_comparison.csv');best=table[table.model==config['selected_population']].iloc[0];baseline=table[table.model=='persistence'].iloc[0];improvement=100*(baseline.patient_mean_mae-best.patient_mean_mae)/baseline.patient_mean_mae
    cs=st.columns(4);cs[0].metric('30-minute MAE',f'{best.patient_mean_mae:.2f} mg/dL');cs[1].metric('Error reduction vs unchanged',f'{improvement:.1f}%');cs[2].metric('New episodes warned',f'{int(best.detected_events)}/{int(best.eligible_events)}');cs[3].metric('False warnings / hour',f'{best.false_warnings_per_hour:.3f}')
    event=results[model_name]['events']['overall'];type2=results[config['selected_population']]['events']['type2']
    st.warning(f"Type 2 subgroup: {type2['detected_events']}/{type2['eligible_events']} eligible onsets credited, with {type2['false_warnings_per_hour']:.3f} false warnings per eligible non-event hour. Only 3 Type 2 test participants. The overall rate combines all cohorts.")
    selected_events=results[config['selected_population']]['events']['overall']
    st.caption(f"Scoring v2: causal warning emission and cooldown. Positive matching uses observed sustained onsets. Unmatched alerts need complete follow-up. {selected_events.get('censored_warnings',0)} emitted warnings have censored outcomes and do not count as false warnings. Thresholds remain frozen from scoring-v1 validation. Corrected evaluation is retrospective; prospective usefulness remains unproven.")
    st.write('')
    display=table[['model','patient_mean_mae','type2_mae','event_sensitivity','false_warnings_per_hour','median_lead_minutes','patient_mean_coverage']].copy();display.columns=['Model','MAE · all cohorts','MAE · Type 2','Event sensitivity','False warnings/hour','Median lead · min','Coverage'];st.dataframe(display,hide_index=True,width='stretch')
    st.info('Clinical fusion did not outperform the CGM-only model on validation. Online bias correction was tested and disabled. The adaptive-twin row therefore equals the selected population model; it is not an independent improvement.')
    left,right=st.columns(2)
    with left:
        st.markdown('**Per-patient error**');patient=pd.read_csv(ROOT/'reports/test_per_patient.csv');subset=patient[patient.model.isin(['persistence',config['selected_population'],'clinical_activity'])]
        fig=go.Figure()
        for name,color in [('persistence','#708198'),(config['selected_population'],'#67e2d0'),('clinical_activity','#9bb5ff')]:
            d=subset[subset.model==name];fig.add_bar(x=[f'P{x:03d}'for x in d.participant],y=d.mae,name=names[name],marker_color=color)
        fig.update_layout(height=300,barmode='group',paper_bgcolor='#0b1220',plot_bgcolor='#0b1220',font_color='#afbed0',margin=dict(t=10,b=5),legend=dict(orientation='h',y=-.25),yaxis_title='30-minute MAE · mg/dL');st.plotly_chart(fig,width='stretch')
    with right:
        st.markdown('**What these numbers mean**');st.write('MAE is the average distance between forecast and later glucose, averaged equally across scoreable participants. Event sensitivity counts eligible new sustained high-glucose onsets warned within 30 minutes.')
        st.write('False warnings are unmatched alerts per eligible non-event hour. Median lead time describes successful warnings only; misses remain in the sensitivity denominator.')
        intervals=[r for r in results['adaptive_twin']['forecast']['per_patient'] if 'interval_coverage' in r];coverage=np.mean([r['interval_coverage'] for r in intervals]);st.write(f'Nominal 90% band: observed patient-mean test coverage **{coverage:.1%}**, width **{2*config["interval_half_widths"][-1]:.1f} mg/dL** at 30 minutes.')
    st.markdown('**Warning outcomes per participant · selected model**')
    st.dataframe(pd.DataFrame(results[config['selected_population']]['events']['per_patient']).drop(columns=['leads']),hide_index=True,width='stretch')
    st.download_button('Download complete comparison',table.to_csv(index=False),'glucotwin_comparison.csv','text/csv')
with tab3:
    st.subheader('A traceable path from data to forecast')
    st.markdown('**Linked clinical profile → first-day cadence initialization → past-hour state → six-step forecast → episode review**')
    audit=json.loads((ROOT/'reports/audit_summary.json').read_text());cols=st.columns(4)
    for c,label,value in zip(cols,['Participants','Type 2 participants','Source rows','Supported windows'],[45,14,f'{audit["raw_rows"]:,}',f'{audit["supported_windows"]:,}']):c.metric(label,str(value))
    st.markdown('**Source and preprocessing**');st.write(audit['preprocessing']);st.write('HbA1c is interpreted as percent, consistent with the source’s cohort definitions; its dictionary unit label is inconsistent. MET values are divided by 10. Meal logs, medication doses, Libre readings and participant IDs are excluded from model features.')
    st.markdown('**Important limits**');st.write('Small US cohort; only three Type 2 participants in the test split. Published interpolation prevents full verification of original device timestamps and missing-reading provenance. These results need replication on raw CGM data and an independent cohort. This prototype does not recommend treatment or simulate causal meal effects.')
    st.dataframe(pd.read_csv(ROOT/'reports/data_audit.csv'),hide_index=True,width='stretch')
    st.markdown('[Original dataset and attribution](https://physionet.org/content/cgmacros/1.0.0/) · Dataset license: CC BY-NC-SA 4.0. Numeric ZIP members were CRC-verified and SHA256 recorded; the full photo archive was not downloaded or checksum-verified.')

with tab4:
    st.subheader('Forecast Reliability Lab')
    st.write('Inspect what the model actually received, how disruption changes forecast availability, and errors that became observable later.')
    st.info('Controlled recorded-data experiment. The delivery scenarios simulate transport failures, not physiology. Recent error is descriptive evidence, not a calibrated probability of correctness.')
    duration=st.select_slider('Experiment duration (hours)',options=[6,12,24],value=12)
    @st.cache_data(show_spinner='Running the same frozen model under three delivery conditions...')
    def lab_runs(participant,engine,hours):
        data=frame[frame.participant==participant].reset_index(drop=True)
        starts=np.flatnonzero(data.supported)
        if not len(starts):return None
        first=int(starts[0]);segment=data.iloc[max(0,first-12):first+hours*12+1]
        records=segment.to_dict('records');base_profile={c:data.iloc[0][c] for c in CLINICAL}
        outputs={}
        for scenario in SCENARIOS:
            rows,ledger=simulate_segment(records,base_profile,models[engine],scenario,config['interval_half_widths'] if engine==config['selected_population'] else None,model_fingerprint)
            outputs[scenario]={'rows':rows,'receipts_json':ledger.export(),'receipts':ledger.receipts,'recent':ledger.summary()}
        return outputs
    runs=lab_runs(int(pid),model_name,duration)
    if runs is None:
        st.info('This participant has no supported forecast windows under the fixed cadence rule. No lab forecasts can be issued.')
    else:
        st.caption(f'P{pid:03d}, {names[model_name]}. Same chronological segment and model for all scenarios. All outcomes in this tab are retrospective research observations.')
        st.write('Signal loss removes 15 minutes every 6 hours. Delayed delivery supplies packets 10 minutes old for 30 minutes every 6 hours. Packets older than 5 minutes fail the freshness rule. We preserve their timestamps, withhold the current forecast, and require new uninterrupted history.')
        summary=[]
        for scenario,run in runs.items():
            rows=run['rows'];values=[r['outcomes'][5]['absolute_error'] for r in run['receipts'] if r['outcomes'][5] and r['outcomes'][5]['absolute_error'] is not None]
            summary.append({'Delivery scenario':SCENARIOS[scenario],'Forecast coverage':float(rows.forecast_30.notna().mean()),'Observed matured +30 forecasts':len(values),'MAE of matured receipts (mg/dL)':float(np.mean(values)) if values else None,'Rejected stale packets':int((rows.delivery_status=='stale_rejected').sum())})
        st.dataframe(pd.DataFrame(summary),hide_index=True,width='stretch')
        st.caption('Scenario MAEs can use different retained targets. A lower error after withholding does not prove better accuracy. Compare availability and inspect the full-cohort paired analysis below.')
        plot=go.Figure()
        for scenario,color in [('normal','#67e2d0'),('dropout','#f3bf6a'),('delay','#9bb5ff')]:
            rows=runs[scenario]['rows'];plot.add_trace(go.Scatter(x=pd.to_datetime(rows.time)+pd.Timedelta(minutes=30),y=rows.forecast_30,name=SCENARIOS[scenario],connectgaps=False,line=dict(color=color,width=2)))
        plot.update_layout(height=380,paper_bgcolor='#0b1220',plot_bgcolor='#0b1220',font_color='#afbed0',legend=dict(orientation='h',y=1.2),margin=dict(t=50),yaxis_title='Issued +30-minute forecast (mg/dL)',xaxis_title='Forecast target time (shifted research dates)')
        st.plotly_chart(plot,width='stretch')
        st.caption('Gaps mean withheld forecasts. Overlapping curves mean identical retained input windows, not independent agreement between models.')
        selected_scenario=st.selectbox('Inspect delivery scenario',list(SCENARIOS),format_func=lambda k:SCENARIOS[k],key='lab_scenario')
        run=runs[selected_scenario]
        st.dataframe(run['rows'][['time','source_time','age_minutes','delivery_status','status','reason','forecast_30']],hide_index=True,width='stretch')
        st.download_button('Download scenario forecast receipts (JSON)',run['receipts_json'],f'P{pid:03d}_{selected_scenario}_receipts.json','application/json',key='lab_receipts')
        if run['receipts']:
            receipt_time=st.selectbox('Inspect an issued forecast receipt',[r['issued_at'] for r in run['receipts']],key=f'lab_receipt_{pid}_{model_name}_{selected_scenario}_{duration}')
            receipt=next(r for r in run['receipts'] if r['issued_at']==receipt_time)
            st.json(receipt,expanded=False)
            st.caption('The receipt hash covers the issued payload and input snapshot, excluding later outcomes. It supports content checking, not authentication or a signed medical record.')
    benchmark_path=ROOT/'reports/reliability_benchmark.json'
    if benchmark_path.exists():
        benchmark=json.loads(benchmark_path.read_text())
        st.markdown('**Full reserved-cohort disruption benchmark**')
        display=pd.DataFrame(benchmark['summary'])[['label','reserved_patients','scoreable_patients','patient_mean_coverage','patient_mean_mae_mg_dl','paired_normal_mae_mg_dl','additional_withheld_rows','stale_packets_rejected']]
        st.dataframe(display,hide_index=True,width='stretch')
        st.caption(benchmark['interpretation'])
        st.download_button('Download complete reliability benchmark',benchmark_path.read_text(),'reliability_benchmark.json','application/json')

st.caption('GlucoTwin / Digital Twin Challenge 2026 / Research prototype · Built for transparent forecasting and clinician review')
