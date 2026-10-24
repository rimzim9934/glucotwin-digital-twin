from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dashboard_replay_loss_recovery_and_fusion():
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/test_replay.joblib').exists():return
    at=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    assert not at.exception
    assert at.metric[3].value=='1'
    at.button[0].click().run();assert at.button[0].label=='Pause'
    at.button[0].click().run();assert at.button[0].label=='▶ Play'
    at.button[3].click().run()
    at.button[2].click().run()
    assert not at.exception and at.metric[3].value=='7'
    at.toggle[1].set_value(True).run();at.button[1].click().run()
    assert not at.exception and at.metric[1].value=='Withheld' and at.metric[0].value=='Unavailable'
    at.toggle[1].set_value(False).run()
    for _ in range(3):at.button[2].click().run()
    assert not at.exception and at.metric[2].value=='Updating'
    at.sidebar.selectbox[1].select('clinical_activity').run()
    assert not at.exception and at.metric[1].value!='Withheld'


def test_reliability_lab_receipt_and_unsupported_participant():
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/test_replay.joblib').exists():return
    at=AppTest.from_file(str(root/'app.py'),default_timeout=45).run()
    assert not at.exception
    assert any(t.label=='Reliability lab' for t in at.tabs)
    scenario=next(x for x in at.selectbox if x.label=='Inspect delivery scenario')
    scenario.select('delay').run()
    assert not at.exception
    assert any('delivery_status' in df.value.columns and df.value.delivery_status.eq('stale_rejected').any() for df in at.dataframe)
    at.sidebar.selectbox[0].select(18).run()
    assert not at.exception
    assert any('No lab forecasts can be issued' in x.value for x in at.info)
