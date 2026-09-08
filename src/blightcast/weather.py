"""Daily weather per district from the raw hourly Open-Meteo pulls.

For each district and calendar day (Europe/London), computes
  tmin, tmax, tmean            2 m air temperature
  rh90h, rh85h, rh95h          hours with RH >= threshold
  wet10h                       hours with RH >= 90 and T >= 10 together
  rain                         mm
  hutton_day = tmin >= 10 and rh90h >= 6      (Dancey, Skelsey & Cooke 2017)
  smith_day  = tmin >= 10 and rh90h >= 11     (Smith 1956)
A Hutton period is two consecutive hutton_days; `hutton_period` marks the second day (the day the
alert is issued). Same for `smith_period`.

Output: data/processed/daily.csv.gz (one row per district-day) and per-district cache in
data/processed/daily/<outcode>.<model>.csv so re-runs only process new pulls.
"""
import glob, gzip, json, os, sys
import numpy as np, pandas as pd
from . import RAW, PROC

def daily_from_file(path):
    d = json.load(gzip.open(path, 'rt'))
    h = pd.DataFrame(d['hourly'])
    h['time'] = pd.to_datetime(h['time'])
    h['day'] = h['time'].dt.floor('D')
    t, rh = h['temperature_2m'], h['relative_humidity_2m']
    h['rh90'] = (rh >= 90).astype(float); h['rh85'] = (rh >= 85).astype(float); h['rh95'] = (rh >= 95).astype(float)
    h['wet10'] = ((rh >= 90) & (t >= 10)).astype(float)
    g = h.groupby('day').agg(tmin=('temperature_2m', 'min'), tmax=('temperature_2m', 'max'),
                             tmean=('temperature_2m', 'mean'), rh90h=('rh90', 'sum'), rh85h=('rh85', 'sum'),
                             rh95h=('rh95', 'sum'), wet10h=('wet10', 'sum'), rain=('precipitation', 'sum'),
                             nhours=('temperature_2m', 'count'))
    g = g[g.nhours >= 20].drop(columns='nhours')
    g = flags(g)
    g.insert(0, 'outcode', d['_outcode'])
    return g.reset_index().rename(columns={'day': 'date'})

def flags(g):
    """Apply the Smith and Hutton rules to a daily frame indexed by date (one location)."""
    g = g.copy()
    g['hutton_day'] = ((g.tmin >= 10) & (g.rh90h >= 6)).astype(int)
    g['smith_day'] = ((g.tmin >= 10) & (g.rh90h >= 11)).astype(int)
    idx_ok = g.index.to_series().diff().dt.days.eq(1)
    g['hutton_period'] = (g.hutton_day.eq(1) & g.hutton_day.shift(1).eq(1) & idx_ok).astype(int)
    g['smith_period'] = (g.smith_day.eq(1) & g.smith_day.shift(1).eq(1) & idx_ok).astype(int)
    return g

def main(model='era5_land'):
    cache = os.path.join(PROC, 'daily'); os.makedirs(cache, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, 'weather', f'*.{model}.json.gz')))
    new = 0
    for f in files:
        oc = os.path.basename(f).split('.')[0]
        out = os.path.join(cache, f'{oc}.{model}.csv')
        if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(f): continue
        daily_from_file(f).to_csv(out, index=False, float_format='%.2f'); new += 1
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(cache, f'*.{model}.csv')))]
    alld = pd.concat(parts, ignore_index=True)
    alld.to_csv(os.path.join(PROC, f'daily.{model}.csv.gz'), index=False)
    print(f'{len(files)} districts pulled, {new} newly processed, {len(alld)} district-days, '
          f'{alld.date.min()} to {alld.date.max()}')
    s = alld[pd.to_datetime(alld.date).dt.month.between(6, 9)]
    print('June-Sept share of days: hutton_day %.2f  hutton_period %.2f  smith_period %.2f' %
          (s.hutton_day.mean(), s.hutton_period.mean(), s.smith_period.mean()))

if __name__ == '__main__':
    main(*sys.argv[1:])
