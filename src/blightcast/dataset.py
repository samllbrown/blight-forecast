"""District-day panel for the season window (1 May - 31 Oct) with features known on the day.

Weather features use days <= t. Report features use reports dated <= t-1 (a report on day t is
the outcome, not a predictor). District base rates use prior seasons only.

Outcomes (all "at least one report ..."):
  y7        in this district, dated in (t, t+7]
  y14       in this district, (t, t+14]
  y7_lag    in this district, (t+7, t+21]      (report lag: today's weather, infections reported 1-3 weeks on)
  yn25_7    within 25 km of this district centroid (including itself), (t, t+7]
  yn25_lag  within 25 km, (t+7, t+21]
  *_crop    crop reports only (conventional, organic, trial), for y7 and yn25_7
"""
import os, sys
import numpy as np, pandas as pd
from . import PROC

SEASON = (5, 10)
EARTH = 6371.0
KERNEL_KM, KERNEL_DAYS, KERNEL_MAXDAYS = 30.0, 10.0, 42

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(a))

def roll_sum(s, n):
    return s.rolling(n, min_periods=1).sum()

def run_length(flag):
    """length of the current run of 1s ending at each position"""
    f = flag.values.astype(int); out = np.zeros(len(f), dtype=int); c = 0
    for i, v in enumerate(f):
        c = c + 1 if v else 0; out[i] = c
    return out

def kernel_series(ords_all, rep_o, rep_w):
    """sum over reports of w * exp(-(t-o)/KERNEL_DAYS) for 1 <= t-o <= KERNEL_MAXDAYS, on the day grid ords_all."""
    base = ords_all[0]; n = len(ords_all); out = np.zeros(n)
    decay = np.exp(-np.arange(1, KERNEL_MAXDAYS + 1) / KERNEL_DAYS)
    for o, w in zip(rep_o, rep_w):
        i0 = o + 1 - base
        if i0 >= n or i0 + KERNEL_MAXDAYS <= 0: continue
        lo, hi = max(i0, 0), min(i0 + KERNEL_MAXDAYS, n)
        out[lo:hi] += w * decay[lo - i0:hi - i0]
    return out

def build(model='era5_land'):
    daily = pd.read_csv(os.path.join(PROC, f'daily.{model}.csv.gz'), parse_dates=['date'])
    ob = pd.read_csv(os.path.join(PROC, 'outbreaks.csv'), parse_dates=['date'])
    dist = pd.read_csv(os.path.join(PROC, 'districts.csv')).set_index('outcode')
    ob['ord'] = ob.date.map(pd.Timestamp.toordinal)
    rep_ord, rep_lat, rep_lon = ob.ord.values, dist.loc[ob.outcode, 'lat'].values, dist.loc[ob.outcode, 'lon'].values
    rep_crop = ob.crop.values.astype(bool)
    season_counts = ob.groupby(['outcode', 'year']).size()
    years = sorted(ob.year.unique())
    rich = 'w_t7' in daily.columns
    frames = []
    for oc, g in daily.groupby('outcode'):
        if oc not in dist.index: continue
        g = g.sort_values('date').reset_index(drop=True)
        # the daily grid must be contiguous for the kernel; reindex to full days
        full = pd.DataFrame({'date': pd.date_range(g.date.min(), g.date.max())}).merge(g, on='date', how='left')
        g = full
        f = pd.DataFrame({'outcode': oc, 'date': g.date})
        hp, hd = g.hutton_period.fillna(0), g.hutton_day.fillna(0)
        f['hutton_alert14'] = roll_sum(hp, 14).gt(0).astype(int)
        f['hutton_alert7'] = roll_sum(hp, 7).gt(0).astype(int)
        f['hutton_n28'] = roll_sum(hp, 28); f['hutton_n14'] = roll_sum(hp, 14)
        f['huttonday_n14'] = roll_sum(hd, 14); f['huttonday_n28'] = roll_sum(hd, 28)
        f['hutton_run'] = run_length(hd)                       # consecutive Hutton days ending today
        f['days_since_hp'] = (g.index.values - pd.Series(np.where(hp.values == 1, g.index.values, np.nan)).ffill().fillna(-999).values).clip(max=120)
        f['smith_alert14'] = roll_sum(g.smith_period.fillna(0), 14).gt(0).astype(int)
        f['smith_n28'] = roll_sum(g.smith_period.fillna(0), 28)
        f['rh90h_7'] = roll_sum(g.rh90h, 7); f['rh90h_14'] = roll_sum(g.rh90h, 14); f['rh90h_28'] = roll_sum(g.rh90h, 28)
        f['rh85h_14'] = roll_sum(g.rh85h, 14); f['rh95h_14'] = roll_sum(g.rh95h, 14)
        f['wet10h_14'] = roll_sum(g.wet10h, 14)
        f['humid_run'] = run_length(g.rh90h.fillna(0) >= 6)   # consecutive days with >= 6 humid hours
        f['tmin_7'] = g.tmin.rolling(7, min_periods=1).mean(); f['tmean_14'] = g.tmean.rolling(14, min_periods=1).mean()
        f['tmax_7'] = g.tmax.rolling(7, min_periods=1).mean()
        f['rain_7'] = roll_sum(g.rain, 7); f['rain_28'] = roll_sum(g.rain, 28)
        f['raindays_14'] = roll_sum((g.rain >= 1).astype(float), 14)
        if rich:
            for c in ('w_t7', 'w_t12', 'w_t16', 'w_t22'):
                f[c + '_14'] = roll_sum(g[c], 14)
            f['w_t12_22_7'] = roll_sum(g.w_t12 + g.w_t16, 7)
            f['wspd_7'] = g.wspd.rolling(7, min_periods=1).mean()
            f['rh_night_7'] = g.rh_night.rolling(7, min_periods=1).mean()
        f['doy'] = g.date.dt.dayofyear; f['year'] = g.date.dt.year
        # thermal time since 1 April (base 4 C): crop-stage proxy
        gdd = (g.tmean.fillna(0) - 4).clip(lower=0)
        f['gdd_apr'] = gdd.where(g.date.dt.month >= 4, 0).groupby(g.date.dt.year).cumsum()
        ords = g.date.map(pd.Timestamp.toordinal).values
        # report-based features
        dkm = hav(dist.loc[oc, 'lat'], dist.loc[oc, 'lon'], rep_lat, rep_lon)
        own = ob.outcode.values == oc
        def count_window(mask, lo, hi):
            r = np.sort(rep_ord[mask])
            return np.searchsorted(r, ords - lo, side='right') - np.searchsorted(r, ords - hi, side='left')
        f['near20_14'] = count_window(dkm <= 20, 1, 14)
        f['near40_14'] = count_window(dkm <= 40, 1, 14)
        f['near40_28'] = count_window(dkm <= 40, 1, 28)
        f['near100_28'] = count_window(dkm <= 100, 1, 28)
        f['own_21'] = count_window(own, 1, 21); f['own_60'] = count_window(own, 1, 60)
        wk = np.exp(-dkm / KERNEL_KM); sel = dkm <= 150
        f['kern_all'] = kernel_series(ords, rep_ord[sel], wk[sel])
        f['kern_crop'] = kernel_series(ords, rep_ord[sel & rep_crop], wk[sel & rep_crop])
        f['kern_noncrop'] = kernel_series(ords, rep_ord[sel & ~rep_crop], wk[sel & ~rep_crop])
        jan1 = pd.to_datetime(f.year.astype(str) + '-01-01').map(pd.Timestamp.toordinal).values
        r_all = np.sort(rep_ord); r_100 = np.sort(rep_ord[dkm <= 100])
        f['nat_ytd'] = np.searchsorted(r_all, ords - 1, side='right') - np.searchsorted(r_all, jan1, side='left')
        f['reg100_ytd'] = np.searchsorted(r_100, ords - 1, side='right') - np.searchsorted(r_100, jan1, side='left')
        sc = season_counts.loc[oc] if oc in season_counts.index.get_level_values(0) else pd.Series(dtype=float)
        prior = {y: (sum(sc.get(yy, 0) for yy in years if yy < y) / max(1, sum(1 for yy in years if yy < y))) for y in f.year.unique()}
        f['prior_rate'] = f.year.map(prior)
        f['lat'] = dist.loc[oc, 'lat']; f['lon'] = dist.loc[oc, 'lon']; f['elev'] = dist.loc[oc, 'elev'] if 'elev' in dist.columns else 0
        # outcomes
        def outcome(mask, lo, hi):
            r = np.sort(rep_ord[mask])
            return (np.searchsorted(r, ords + hi, side='right') - np.searchsorted(r, ords + lo, side='right') > 0).astype(int)
        n25 = dkm <= 25
        f['y7'] = outcome(own, 0, 7); f['y14'] = outcome(own, 0, 14); f['y7_lag'] = outcome(own, 7, 21)
        f['yn25_7'] = outcome(n25, 0, 7); f['yn25_lag'] = outcome(n25, 7, 21)
        f['y7_crop'] = outcome(own & rep_crop, 0, 7); f['yn25_7_crop'] = outcome(n25 & rep_crop, 0, 7)
        f['y0'] = (np.searchsorted(np.sort(rep_ord[own]), ords, side='right') - np.searchsorted(np.sort(rep_ord[own]), ords, side='left') > 0).astype(int)
        f = f[f.date.dt.month.between(*SEASON) & g.tmin.notna().values]
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel['region'] = panel.outcode.map(dist.country)
    panel['week'] = panel.doy // 7
    # week-of-year climatology from prior seasons only, per outcome family
    overall = panel.y7.mean()
    for yname, cname in [('y7', 'clim_week'), ('yn25_7', 'clim_week_n25')]:
        wk = panel.groupby(['year', 'week'])[yname].agg(['sum', 'size']).reset_index()
        clim = {}
        for y in sorted(panel.year.unique()):
            prev = wk[wk.year < y]
            if prev.empty: clim[y] = None; continue
            g2 = prev.groupby('week').agg(s=('sum', 'sum'), n=('size', 'sum'))
            clim[y] = ((g2.s + 1) / (g2.n + 20)).to_dict()
        ov = panel[yname].mean()
        panel[cname] = [clim[y][w] if clim[y] and w in clim[y] else ov for y, w in zip(panel.year, panel.week)]
    return panel

def main(model='era5_land'):
    p = build(model)
    p.to_csv(os.path.join(PROC, f'panel.{model}.csv.gz'), index=False, float_format='%.3f')
    print(f'{len(p)} district-days, {p.outcode.nunique()} districts, seasons {p.year.min()}-{p.year.max()}, '
          f'rates y7 {p.y7.mean():.4f} y7_lag {p.y7_lag.mean():.4f} yn25_7 {p.yn25_7.mean():.4f} yn25_lag {p.yn25_lag.mean():.4f}, '
          f'hutton_alert14 on {p.hutton_alert14.mean():.3f}')

if __name__ == '__main__':
    main(*sys.argv[1:])
