"""Pull Meteostat bulk hourly files for every GB station with hourly data covering the study years.
No registration. Files: https://bulk.meteostat.net/v2/hourly/<id>.csv.gz
Columns: date,hour,temp,dwpt,rhum,prcp,snow,wdir,wspd,wpgt,pres,tsun,coco
Station list saved to data/raw/meteostat/stations.csv.
"""
import gzip, json, os, csv, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data/raw/meteostat'); os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (blight-forecast research)'}
def get(url, path):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                open(path, 'wb').write(r.read()); return True
        except Exception as e:
            print('  retry', url, e); time.sleep(5 * (attempt + 1))
    return False
sp = os.path.join(OUT, 'stations_full.json.gz')
if not os.path.exists(sp): get('https://bulk.meteostat.net/v2/stations/full.json.gz', sp)
st = json.load(gzip.open(sp))
keep = []
for s in st:
    if s['country'] not in ('GB', 'IM', 'JE', 'GG'): continue
    inv = s['inventory']['hourly']
    if not inv['start'] or not inv['end']: continue
    if inv['start'] > '2012-01-01' or inv['end'] < '2020-01-01': continue
    keep.append(dict(id=s['id'], name=s['name']['en'], wmo=s['identifiers'].get('wmo'), icao=s['identifiers'].get('icao'),
                     lat=s['location']['latitude'], lon=s['location']['longitude'], elev=s['location']['elevation'],
                     start=inv['start'], end=inv['end']))
with open(os.path.join(OUT, 'stations.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(keep[0])); w.writeheader(); w.writerows(keep)
print(len(keep), 'stations', flush=True)
for i, s in enumerate(keep):
    p = os.path.join(OUT, f"{s['id']}.csv.gz")
    if os.path.exists(p) and os.path.getsize(p) > 0: continue
    ok = get(f"https://bulk.meteostat.net/v2/hourly/{s['id']}.csv.gz", p)
    if i % 10 == 0: print(i, s['id'], s['name'], ok, flush=True)
    time.sleep(0.3)
print('done', flush=True)
