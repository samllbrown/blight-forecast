"""Pull hourly ERA5-Land (via Open-Meteo archive API) for every postcode district
in the Fight Against Blight record, 2006-01-01 to the latest available date.

Stdlib only so it can run without the project venv. Raw hourly JSON is stored
gzipped in data/raw/weather/<outcode>.json.gz (about 1 MB each); daily
aggregates are computed later by blightcast.weather.

Usage: python3 scripts/pull_weather.py [--model era5_land] [--sleep 1.0]
Re-runnable: districts already on disk are skipped.
"""
import argparse, glob, gzip, json, os, sys, time, urllib.request, urllib.error, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAB = os.path.join(ROOT, 'data/raw/fab')
OUT = os.path.join(ROOT, 'data/raw/weather')

def centroids():
    """Modal centroid per outcode from the outbreak record."""
    per = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(os.path.join(FAB, '20*.json'))):
        for r in json.load(open(f)):
            oc, lat, lon = r.get('outcode'), r.get('realLatitude'), r.get('realLongitude')
            if oc and lat and lon:
                per[oc][(round(lat, 4), round(lon, 4))] += 1
    return {oc: c.most_common(1)[0][0] for oc, c in per.items()}, {oc: sum(c.values()) for oc, c in per.items()}

def pull(lat, lon, model, end):
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date=2006-01-01&end_date={end}&hourly=temperature_2m,relative_humidity_2m,precipitation"
           f"&models={model}&timezone=Europe/London")
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read()[:120].decode(errors='replace')
            # 429 carries minutely / hourly / daily wording; wait accordingly and never give up
            wait = 65 if 'inutely' in body else 600 if 'ourly' in body else 3600 if 'aily' in body else 30
            if attempt % 10 == 0: print(f'  HTTP {e.code} {body}, sleeping {wait}s', flush=True)
            time.sleep(wait); attempt += 1
        except Exception as e:
            print(f'  {e!r}, sleeping 15s', flush=True); time.sleep(15)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='era5_land')
    ap.add_argument('--sleep', type=float, default=1.0)
    ap.add_argument('--end', default=time.strftime('%Y-%m-%d', time.gmtime(time.time() - 7 * 86400)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cents, counts = centroids()
    # busiest districts first so partial downloads are useful early
    todo = [oc for oc in sorted(cents, key=lambda o: (-counts[o], o)) if not os.path.exists(os.path.join(OUT, f'{oc}.{a.model}.json.gz'))]
    print(f'{len(cents)} districts, {len(todo)} to pull, end {a.end}', flush=True)
    t0 = time.time()
    for i, oc in enumerate(todo):
        lat, lon = cents[oc]
        d = pull(lat, lon, a.model, a.end)
        d['_outcode'] = oc; d['_requested'] = [lat, lon]
        tmp = os.path.join(OUT, f'{oc}.{a.model}.json.gz.part')
        with gzip.open(tmp, 'wt') as f: json.dump(d, f)
        os.replace(tmp, tmp[:-5])
        if i % 10 == 0:
            n = len(d['hourly']['time']); print(f'{i+1}/{len(todo)} {oc} {n} hours, {time.time()-t0:.0f}s', flush=True)
        time.sleep(a.sleep)
    print('done', flush=True)

if __name__ == '__main__':
    main()
