"""Export everything the blog post needs into one JSON (scratch/blight/blog.json in the blog repo).
Run: PYTHONPATH=src .venv/bin/python scripts/export_blog.py
"""
import os, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from blightcast import PROC, RES, RAW
OUT = '/home/sam/samuellbrown.dev/scratch/blight/blog.json'
DEMO = ['DD8', 'IP12', 'CT3', 'SA48', 'TF10']
NAMES = {'DD8': 'Angus (Forfar)', 'IP12': 'Suffolk (Woodbridge)', 'CT3': 'Kent (Canterbury)', 'SA48': 'Ceredigion (Lampeter)', 'TF10': 'Shropshire (Newport)'}
ANIM_YEARS = [2012, 2018, 2024]

daily = pd.read_csv(os.path.join(PROC, 'daily.stations.csv.gz'), parse_dates=['date'])
ob = pd.read_csv(os.path.join(PROC, 'outbreaks.csv'), parse_dates=['date'])
dist = pd.read_csv(os.path.join(PROC, 'districts.csv'))
# add the busiest Cornwall and Aberdeenshire districts
for pref in ('TR', 'AB'):
    top = ob[ob.outcode.str.startswith(pref)].groupby('outcode').size().idxmax(); DEMO.append(top)
NAMES.setdefault(DEMO[-2], 'Cornwall (' + DEMO[-2] + ')'); NAMES.setdefault(DEMO[-1], 'Aberdeenshire (' + DEMO[-1] + ')')
season = daily[daily.date.dt.month.between(5, 10)]
out = {'demo': {}, 'names': NAMES, 'order': DEMO}
# per demo district: per season arrays (May 1 .. Oct 31 = 184 days), tmin*10 int, rh90h int, reports as day offsets (+crop flag)
for oc in DEMO:
    d = season[season.outcode == oc]
    yrs = {}
    for y, g in d.groupby(d.date.dt.year):
        g = g.set_index('date').reindex(pd.date_range(f'{y}-05-01', f'{y}-10-31'))
        if g.tmin.isna().mean() > 0.2: continue
        rep = ob[(ob.outcode == oc) & (ob.year == y)]
        offs = [(int((r.date - pd.Timestamp(f'{y}-05-01')).days), int(r.crop)) for r in rep.itertuples() if 0 <= (r.date - pd.Timestamp(f'{y}-05-01')).days < 184]
        yrs[int(y)] = {'tmin': [None if pd.isna(v) else int(round(v * 10)) for v in g.tmin],
                       'rh': [None if pd.isna(v) else int(v) for v in g.rh90h],
                       'reports': offs}
    out['demo'][oc] = yrs
# model predictions + inputs for the demo districts (2012-2025) from the saved preds and panel
pr = pd.read_csv(os.path.join(RES, 'tables', 'preds.stations.y7.csv.gz'), parse_dates=['date'])
panel_cols = ['outcode', 'date', 'clim_week', 'kern_all', 'prior_rate', 'huttonday_n14', 'w_t12_14', 'w_t16_14', 'near100_28']
pan = pd.read_csv(os.path.join(PROC, 'panel.stations.csv.gz'), usecols=panel_cols, parse_dates=['date'])
pan = pan[pan.outcode.isin(DEMO)]
pr = pr[pr.outcode.isin(DEMO)][['outcode', 'date', 'p_gbm_all', 'p_gbm_weather', 'hutton_alert14']].merge(pan, on=['outcode', 'date'])
out['model'] = {}
for oc, g in pr.groupby('outcode'):
    out['model'][oc] = {}
    for y, gy in g.groupby(g.date.dt.year):
        gy = gy.set_index('date').reindex(pd.date_range(f'{y}-05-01', f'{y}-10-31'))
        f = lambda col, k: [None if pd.isna(v) else round(float(v), k) for v in gy[col]]
        out['model'][oc][int(y)] = {'p': f('p_gbm_all', 3), 'pw': f('p_gbm_weather', 3), 'kern': f('kern_all', 2), 'hd14': f('huttonday_n14', 0),
                                    'prior': round(float(gy.prior_rate.dropna().iloc[0]), 2) if gy.prior_rate.notna().any() else None}
# week-of-year climatology is the same for every district within a season: store once per season (index = day offset // 7)
out['clim'] = {}
for y, gy in pr.groupby(pr.date.dt.year):
    gy = gy[gy.outcode == DEMO[0]].set_index('date').reindex(pd.date_range(f'{y}-05-01', f'{y}-10-31'))
    out['clim'][int(y)] = [None if pd.isna(v) else round(float(v), 4) for v in gy.clim_week]
# curves, groups, weeks, part_b tables
cur = pd.read_csv(os.path.join(RES, 'tables', 'curves.stations.y7.csv'))
out['curves'] = {m: [[round(r.alert_rate, 3), round(r.recall, 3)] for r in g.dropna().itertuples()] for m, g in cur.groupby('model', sort=False)}
grp = pd.read_csv(os.path.join(RES, 'tables', 'groups.stations.y7.csv'))
out['groups'] = grp.round(3).to_dict('records')
wk = pd.read_csv(os.path.join(RES, 'tables', 'by_week.stations.y7.csv'))
out['weeks'] = wk.round(3).to_dict('records')
out['part_b'] = {}
for y in ['y7', 'y7_lag', 'yn25_7', 'yn25_lag']:
    t = pd.read_csv(os.path.join(RES, 'tables', f'part_b.stations.{y}.csv'))
    out['part_b'][y] = t.drop(columns=['per_season'], errors='ignore').round(3).to_dict('records')
out['part_a'] = json.load(open(os.path.join(RES, 'tables', 'part_a.stations.json')))
# map: every district dot with scores; animation flags for three seasons
ds = pd.read_csv(os.path.join(RES, 'tables', 'district_scores.stations.y7.csv')).set_index('outcode')
counts = ob.groupby('outcode').size()
out['districts'] = [{'oc': r.outcode, 'lat': round(r.lat, 3), 'lon': round(r.lon, 3), 'n': int(counts.get(r.outcode, 0)), 'country': r.country if isinstance(r.country, str) else '',
                     'hut': round(float(ds.hutton_auc[r.outcode]), 3) if r.outcode in ds.index else None,
                     'mod': round(float(ds.best_auc[r.outcode]), 3) if r.outcode in ds.index else None} for r in dist.itertuples()]
out['anim'] = {}
for y in ANIM_YEARS:
    d = season[season.date.dt.year == y]
    idx = pd.date_range(f'{y}-05-01', f'{y}-10-31')
    flags = {}
    for oc, g in d.groupby('outcode'):
        g = g.set_index('date').reindex(idx)
        hp = g.hutton_period.fillna(0).astype(int)
        alert = hp.rolling(14, min_periods=1).sum().gt(0).astype(int)   # same definition as the panel's hutton_alert14
        flags[oc] = ''.join(map(str, alert.values))
    reps = [[r.outcode, int((r.date - idx[0]).days)] for r in ob[ob.year == y].itertuples() if 0 <= (r.date - idx[0]).days < 184]
    out['anim'][y] = {'alert': flags, 'reports': reps}
# coastline
from shapely.geometry import shape
from shapely.ops import unary_union
geo = json.load(open('/home/sam/fluke-forecast/data/raw/geo/countries.geojson'))
rings = []
for ft in geo['features']:
    g = shape(ft['geometry']).simplify(0.025, preserve_topology=True)
    polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    for p in polys:
        if p.area < 0.02: continue
        rings.append([[round(x, 2), round(y, 2)] for x, y in p.exterior.coords])
out['coast'] = rings
json.dump(out, open(OUT, 'w'), separators=(',', ':'))
print('written', OUT, os.path.getsize(OUT) // 1024, 'KB; demo districts', DEMO, 'seasons per district', {oc: len(v) for oc, v in out['demo'].items()}, 'coast rings', len(rings), 'points', sum(len(r) for r in rings))
