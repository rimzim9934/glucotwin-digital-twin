"""Retrieve only numeric members from the official, pinned ZIP using HTTP ranges.
Member CRC32 and SHA256 verified/recorded; full archive SHA256 is not claimed.
"""
from pathlib import Path
import concurrent.futures, hashlib, json, struct, urllib.request, zlib, time, ssl, certifi
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'
URL='https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip'
def fetch(url, start=None, end=None):
    headers={} if start is None else {'Range':f'bytes={start}-{end}'}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90, context=ssl.create_default_context(cafile=certifi.where())) as r:
                if start is not None and r.status!=206: raise ValueError('Server did not honor range request')
                return r.read()
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
def main():
    RAW.mkdir(parents=True,exist_ok=True)
    tail=fetch(URL,657121804,657187339)
    e=tail.rfind(b'PK\x05\x06'); record=struct.unpack('<4s4H2IH',tail[e:e+22]); size,offset=record[5:7]
    cached=Path('/tmp/gluco-zip-index')
    directory=cached.read_bytes() if cached.exists() and cached.stat().st_size==size+22 else fetch(URL,offset,offset+size-1)
    entries=[]; pos=0
    while directory[pos:pos+4]==b'PK\x01\x02':
        x=struct.unpack('<4s6H3I5H2I',directory[pos:pos+46]); n,e,c=x[10:13]
        name=directory[pos+46:pos+46+n].decode('utf-8'); pos+=46+n+e+c
        if name.endswith('.csv') and '__MACOSX' not in name and '/._' not in name:
            entries.append(dict(name=name,method=x[4],crc=x[7],compressed=x[8],size=x[9],offset=x[16],name_length=n,extra_length=e))
    def member(d):
        dest=RAW/'records'/d['name']; dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and zlib.crc32(dest.read_bytes())==d['crc']: b=dest.read_bytes()
        else:
            raw=fetch(URL,d['offset'],d['offset']+30+d['name_length']+d['extra_length']+d['compressed']+1024)
            h=struct.unpack('<4s5H3I2H',raw[:30]); begin=30+h[-2]+h[-1]; payload=raw[begin:begin+d['compressed']]
            b=zlib.decompress(payload,-15) if d['method']==8 else payload
            assert len(b)==d['size'] and zlib.crc32(b)==d['crc'], d['name']
            dest.write_bytes(b)
        print('Verified',d['name'],len(b),flush=True)
        return {**d,'sha256':hashlib.sha256(b).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: files=list(pool.map(member,entries))
    for name in ['DataDictionary_Bio.csv','DataDictionary_CGMacros-00X.csv','SHA256SUMS.txt']:
        (RAW/name).write_bytes(fetch('https://physionet.org/files/cgmacros/1.0.0/'+name))
    manifest={'source':URL,'version':'1.0.0','license':'CC BY-NC-SA 4.0','retrieval':'HTTP range extraction of numeric CSV members; ZIP CRC32 checked; member SHA256 recorded. Whole archive checksum NOT verified.','expected_archive_sha256':'05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d','files':files}
    (ROOT/'reports/data_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',len(files),'numeric files',flush=True)
if __name__=='__main__':main()
