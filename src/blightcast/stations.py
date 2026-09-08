"""Station-based daily weather for every district, mirroring Blightwatch's method: daily minimum
temperature and daily hours at RH >= 90 % are computed at each station, then interpolated to the
district centroid. Interpolation is inverse-distance-squared weighting over the K nearest stations
within MAXKM that reported that day, with tmin lapse-corrected to district elevation... (no district
elevation is known, so no lapse correction; noted in README). The rule flags are then applied to the
interpolated daily values, as Dancey et al. did.

Input: data/raw/meteostat/<id>.csv.gz (hourly: date,hour,temp,dwpt,rhum,prcp,...)
Output: data/processed/daily.stations.csv.gz in the same schema as daily.era5_land.csv.gz, plus
data/processed/station_daily.csv.gz (per-station daily values, for calibration against ERA5-Land).
"""
import glob, os, sys
import numpy as np, pandas as pd
from . import RAW, PROC
from .weather import flags
from .dataset import hav

K, MAXKM = 4, 80
COLS = ['date', 'hour', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']

def station_daily(path, start='2005-12-01'):
    h = pd.read_csv(path, names=COLS, usecols=['date', 'hour', 'temp', 'rhum', 'prcp', 'wspd'])
    h = h[h.date >= start]
    if h.empty: return None
    h = h.dropna(subset=['temp', 'rhum'])
    h['rh90'] = (h.rhum >= 90).astype(float); h['rh85'] = (h.rhum >= 85).astype(float); h['rh95'] = (h.rhum >= 95).astype(float)
    h['wet10'] = ((h.rhum >= 90) & (h.temp >= 10)).astype(float)
    # humid hours by temperature band (Simcast-style inputs; the model learns the weights)
    wet = h.rhum >= 90
    h['w_t7'] = (wet & (h.temp >= 7) & (h.temp < 12)).astype(float)
    h['w_t12'] = (wet & (h.temp >= 12) & (h.temp < 16)).astype(float)
    h['w_t16'] = (wet & (h.temp >= 16) & (h.temp < 22)).astype(float)
    h['w_t22'] = (wet & (h.temp >= 22)).astype(float)
    h['night'] = h.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(float)
    h['rh_night'] = h.rhum * h.night
    g = h.groupby('date').agg(tmin=('temp', 'min'), tmax=('temp', 'max'), tmean=('temp', 'mean'), rh90h=('rh90', 'sum'),
                              rh85h=('rh85', 'sum'), rh95h=('rh95', 'sum'), wet10h=('wet10', 'sum'), rain=('prcp', 'sum'),
                              w_t7=('w_t7', 'sum'), w_t12=('w_t12', 'sum'), w_t16=('w_t16', 'sum'), w_t22=('w_t22', 'sum'),
                              wspd=('wspd', 'mean'), rh_night=('rh_night', 'sum'), night=('night', 'sum'),
                              nhours=('temp', 'count'))
    g = g[g.nhours >= 20].drop(columns='nhours')
    g['rh_night'] = g.rh_night / g.night.replace(0, np.nan); g = g.drop(columns='night')
    g.index = pd.to_datetime(g.index)
    return g

def main():
    st = pd.read_csv(os.path.join(RAW, 'meteostat', 'stations.csv'), dtype={'id': str})
    parts = []
    for _, s in st.iterrows():
        p = os.path.join(RAW, 'meteostat', f'{s.id}.csv.gz')
        if not os.path.exists(p) or os.path.getsize(p) == 0: continue
        g = station_daily(p)
        if g is None or len(g) < 365: continue
        g.insert(0, 'station', s.id); parts.append(g.reset_index().rename(columns={'index': 'date'}))
    sd = pd.concat(parts, ignore_index=True)
    sd.to_csv(os.path.join(PROC, 'station_daily.csv.gz'), index=False, float_format='%.2f')
    summer = sd[sd.date.dt.month.between(6, 9)]
    print(f'{sd.station.nunique()} stations with daily data; days per station median {sd.groupby("station").size().median():.0f}; '
          f'summer hutton_day share at stations {((summer.tmin >= 10) & (summer.rh90h >= 6)).mean():.3f}')
    # interpolate to districts
    dist = pd.read_csv(os.path.join(PROC, 'districts.csv'))
    st = st[st.id.isin(sd.station.unique())].reset_index(drop=True)
    vals = ['tmin', 'tmax', 'tmean', 'rh90h', 'rh85h', 'rh95h', 'wet10h', 'rain', 'w_t7', 'w_t12', 'w_t16', 'w_t22', 'wspd', 'rh_night']
    wide = {v: sd.pivot(index='date', columns='station', values=v).reindex(columns=st.id) for v in vals}
    dates = wide['tmin'].index
    out = []
    for _, d in dist.iterrows():
        dk = hav(d.lat, d.lon, st.lat.values, st.lon.values)
        order = np.argsort(dk); near = order[dk[order] <= MAXKM][:K + 4]   # candidate pool, K used per day after dropping missing
        if len(near) == 0: near = order[:K]
        w_all = 1.0 / np.maximum(dk[near], 2.0) ** 2
        frame = pd.DataFrame(index=dates)
        for v in vals:
            m = wide[v].values[:, near]                      # days x candidates
            ok = ~np.isnan(m)
            # keep the K nearest available per day
            rank = np.cumsum(ok, axis=1); use = ok & (rank <= K)
            w = np.where(use, w_all[None, :], 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                frame[v] = np.nansum(np.where(use, m, 0.0) * w, axis=1) / w.sum(axis=1)
        frame['nst'] = (~np.isnan(wide['tmin'].values[:, near])).sum(axis=1).clip(max=K)
        frame['dist_km'] = dk[near][0]
        frame = frame.dropna(subset=['tmin', 'rh90h'])
        frame = flags(frame)
        frame.insert(0, 'outcode', d.outcode)
        out.append(frame.reset_index().rename(columns={'index': 'date'}))
    alld = pd.concat(out, ignore_index=True)
    alld.to_csv(os.path.join(PROC, 'daily.stations.csv.gz'), index=False, float_format='%.2f')
    s = alld[alld.date.dt.month.between(6, 9)]
    print(f'{alld.outcode.nunique()} districts, {len(alld)} district-days, nearest station median {alld.groupby("outcode").dist_km.first().median():.0f} km; '
          f'June-Sept share: hutton_day {s.hutton_day.mean():.2f} hutton_period {s.hutton_period.mean():.2f} smith_period {s.smith_period.mean():.2f}')

if __name__ == '__main__':
    main()
