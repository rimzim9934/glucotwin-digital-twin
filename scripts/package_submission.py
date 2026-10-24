"""Build a GlucoTwin-only upload archive and verify README artifact links."""
from pathlib import Path
import hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/submission'
PREFIX='TwinPulse_VIT_Bhopal_VIT_Vellore'

def main():
    paths=[ROOT/name for name in ['README.md','START_HERE.md','LICENSE','requirements.txt','.gitignore','app.py','core.py','reliability.py','evaluation.py','prepare.py','train.py','run.py','.streamlit/config.toml','scripts/download_data.py','scripts/check.py','scripts/package_submission.py','scripts/rescore_events.py','scripts/benchmark_reliability.py']]
    for directory in ['docs','tests','reports']:
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json','.csv'}]
    paths += [ROOT/name for name in ['output/pdf/GlucoTwin_Reliability_Architecture.pdf','output/pdf/GlucoTwin_Project_Reliability_Edition.pdf','output/presentation/GlucoTwin_TwinPulse_Reliability_Edition.pptx','output/dashboard.png','output/evidence/example_forecast_receipt.json']]
    paths=sorted(set(paths))
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    selected={p.relative_to(ROOT).as_posix() for p in paths}
    for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
        if target.startswith(('http://','https://','#')):continue
        if target.split('#')[0] not in selected:raise ValueError(f'README target missing from upload package: {target}')
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'GlucoTwin_TwinPulse_Final_Submission.zip'
    entries=[]
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            name=p.relative_to(ROOT).as_posix();archive.write(p,f'{PREFIX}/{name}')
            entries.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        archive.writestr(f'{PREFIX}/PACKAGE_MANIFEST.json',json.dumps({'edition':'Reliability Lab / scoring v2','files':entries},indent=2))
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        assert not any(any(part in {'artifacts','tmp','__pycache__','raw','processed'} for part in Path(n).parts) for n in archive.namelist())
    manifest={'audit_date':'2026-10-24','package':dest.name,'file_count':len(entries)+1,'readme_links_pass':True,'archive_crc_pass':True,'public_access_verified':False,'video_url_present':False,'files':entries}
    (OUT/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

if __name__=='__main__':main()
