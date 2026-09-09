"""The daily service: today's model probability for every district, next to the Hutton alert.

Two commands:

  python -m blightcast.live fit
      Fit the gradient-boosting model with the full feature set on every season in the panel
      (2006 to 2025) and save it with the context it needs at run time: the week-of-year
      climatology from all those seasons, each district's prior-season report rate, and the
      probability thresholds at which the model is on for 30 % and 61 % of test days.
      -> results/models/live_gbm_all.joblib, results/models/live_context.json

  python -m blightcast.live run
      Pull this season's reports from the Fight Against Blight API and weather for every district
      from Open-Meteo (a season cache from the historical-forecast archive, refreshed daily from
      the forecast API with 14 days back and 8 ahead), build the same features as the research
      panel through blightcast.dataset.district_features, score every district for today and the
      week ahead, and write live/latest.json for the blog page. Runs in GitHub Actions each morning.

The weather here is Open-Meteo's model analysis and forecast, not the station interpolation the
model was fitted on; the research found weather adds about one AUC point on top of calendar,
reports and place, so the mismatch moves the probabilities little, but the Hutton flag computed
here will differ from BlightSpy's on some days.
"""
import os, sys, json, time, gzip
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import requests
from . import PROC, RES, ROOT
from .dataset import district_features, hav
from .weather import flags
from .evaluate import SETS, MODELS, NEG_FRAC

MODEL_DIR = os.path.join(RES, 'models')
LIVE_DATA = os.path.join(ROOT, 'data', 'live')
LIVE_OUT = os.path.join(ROOT, 'live')
DEMO = ['DD8', 'IP12', 'CT3', 'SA48', 'TF10', 'TR12', 'AB53']
FAB = 'https://blight.hutton.ac.uk/api/outbreaks?year={year}'
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) blight-forecast/1.0 (+https://github.com/samllbrown/blight-forecast)'}
HOURLY = 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m'
FORECAST_DAYS, PAST_DAYS = 8, 14
CROP = {'Conventional crop', 'Organic crop', 'Trial'}

def log(*a): print(*a, flush=True)

# ---- fit -----------------------------------------------------------------------------
def fit(model='stations'):
    cols = SETS['all']
    usecols = cols + ['outcode', 'date', 'year', 'week', 'y7']
    panel = pd.read_csv(os.path.join(PROC, f'panel.{model}.csv.gz'), usecols=usecols, dtype={c: 'float32' for c in cols})
    log(f'panel {len(panel)} rows, seasons {panel.year.min()}-{panel.year.max()}')
    rng = np.random.default_rng(0)
    keep = (panel.y7.values == 1) | (rng.random(len(panel)) < NEG_FRAC)
    tr = panel[keep]; w = np.where(tr.y7.values == 1, 1.0, 1.0 / NEG_FRAC)
    m = MODELS['gbm'](); t0 = time.time()
    m.fit(tr[cols].values, tr.y7.values, sample_weight=w)
    log(f'fitted on {len(tr)} rows ({int(tr.y7.sum())} positives) in {time.time() - t0:.0f}s')
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'model': m, 'columns': cols}, os.path.join(MODEL_DIR, 'live_gbm_all.joblib'), compress=3)
    # context: climatology by week over all seasons, prior rate per district, thresholds from the held-out predictions
    wk = panel.groupby('week').y7.agg(['sum', 'size'])
    clim = {int(k): float((r['sum'] + 1) / (r['size'] + 20)) for k, r in wk.iterrows()}
    ob = pd.read_csv(os.path.join(PROC, 'outbreaks.csv'), parse_dates=['date'])
    last = int(panel.year.max()); ob = ob[ob.year <= last]        # prior-season rate for next season: reports per fitted season
    years = sorted(ob.year.unique())
    prior = (ob.groupby('outcode').size() / len(years)).round(4).to_dict()
    pr = pd.read_csv(os.path.join(RES, 'tables', f'preds.{model}.y7.csv.gz'), usecols=['p_gbm_all', 'hutton_alert14', 'y7'])
    hut_rate = float(pr.hutton_alert14.mean())
    ctx = {'fitted_seasons': [int(panel.year.min()), int(panel.year.max())], 'clim_week': clim, 'prior_rate': prior, 'prior_seasons': len(years),
           'base_rate': float(pr.y7.mean()), 'hutton_alert_rate': hut_rate,
           'threshold_30': float(np.quantile(pr.p_gbm_all, 0.70)), 'threshold_hutton_rate': float(np.quantile(pr.p_gbm_all, 1 - hut_rate)),
           'threshold_10': float(np.quantile(pr.p_gbm_all, 0.90))}
    json.dump(ctx, open(os.path.join(MODEL_DIR, 'live_context.json'), 'w'), indent=1)
    log('context:', {k: v for k, v in ctx.items() if k not in ('clim_week', 'prior_rate')})
    log('DONE fit')

# ---- run: data pulls ------------------------------------------------------------------
def get_json(url, tries=6):
    for i in range(tries):
        r = requests.get(url, headers=UA, timeout=120)
        if r.status_code == 200: return r.json()
        wait = 20 * (i + 1)
        log(f'  {r.status_code} from {url[:80]}..., waiting {wait}s'); time.sleep(wait)
    raise RuntimeError(f'gave up on {url[:120]}')

def fetch_reports(year):
    os.makedirs(LIVE_DATA, exist_ok=True)
    rows = []
    for y in (year - 1, year):
        path = os.path.join(LIVE_DATA, f'fab_{y}.json')
        if y == year or not os.path.exists(path):
            data = get_json(FAB.format(year=y)); json.dump(data, open(path, 'w'))
        else: data = json.load(open(path))
        rows += data
    df = pd.DataFrame(rows)
    df = df[df.status.eq('confirmed') & df.host.eq('potato')].dropna(subset=['outcode', 'realLatitude', 'realLongitude'])
    df = df[df.outcode.ne('')]
    df['date'] = pd.to_datetime(df.dateSubmitted)
    df = df[~((df.date.dt.month <= 2) | ((df.date.dt.month == 3) & (df.date.dt.day <= 15)))]
    df['crop'] = df.sourceName.isin(CROP).astype(int)
    return df[['outcode', 'date', 'realLatitude', 'realLongitude', 'crop', 'country', 'sourceName']].rename(columns={'realLatitude': 'lat', 'realLongitude': 'lon'}).sort_values('date').reset_index(drop=True)

def hourly_to_daily(block, oc):
    h = pd.DataFrame(block['hourly']); h['time'] = pd.to_datetime(h.time); h['day'] = h.time.dt.floor('D')
    h = h.dropna(subset=['temperature_2m', 'relative_humidity_2m'])
    t, rh = h.temperature_2m, h.relative_humidity_2m; wet = rh >= 90
    h['rh90'] = wet.astype(float); h['rh85'] = (rh >= 85).astype(float); h['rh95'] = (rh >= 95).astype(float)
    h['wet10'] = (wet & (t >= 10)).astype(float)
    h['w_t7'] = (wet & (t >= 7) & (t < 12)).astype(float); h['w_t12'] = (wet & (t >= 12) & (t < 16)).astype(float)
    h['w_t16'] = (wet & (t >= 16) & (t < 22)).astype(float); h['w_t22'] = (wet & (t >= 22)).astype(float)
    h['night'] = h.time.dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(float); h['rh_night'] = rh * h.night
    g = h.groupby('day').agg(tmin=('temperature_2m', 'min'), tmax=('temperature_2m', 'max'), tmean=('temperature_2m', 'mean'),
                             rh90h=('rh90', 'sum'), rh85h=('rh85', 'sum'), rh95h=('rh95', 'sum'), wet10h=('wet10', 'sum'),
                             rain=('precipitation', 'sum'), w_t7=('w_t7', 'sum'), w_t12=('w_t12', 'sum'), w_t16=('w_t16', 'sum'),
                             w_t22=('w_t22', 'sum'), wspd=('wind_speed_10m', 'mean'), rh_night=('rh_night', 'sum'), night=('night', 'sum'),
                             nhours=('temperature_2m', 'count'))
    g = g[g.nhours >= 20].drop(columns='nhours')
    g['rh_night'] = g.rh_night / g.night.replace(0, np.nan); g = g.drop(columns='night')
    g.insert(0, 'outcode', oc)
    return g.reset_index().rename(columns={'day': 'date'})

def pull_weather(dist, base, params, batch=25, pause=1.0):
    parts = []
    ocs = list(dist.index)
    for i in range(0, len(ocs), batch):
        chunk = ocs[i:i + batch]
        lat = ','.join(f'{dist.loc[o, "lat"]:.4f}' for o in chunk); lon = ','.join(f'{dist.loc[o, "lon"]:.4f}' for o in chunk)
        data = get_json(f'{base}?latitude={lat}&longitude={lon}&hourly={HOURLY}&timezone=Europe%2FLondon&{params}')
        if isinstance(data, dict): data = [data]
        for oc, block in zip(chunk, data): parts.append(hourly_to_daily(block, oc))
        log(f'  weather {min(i + batch, len(ocs))}/{len(ocs)}'); time.sleep(pause)
    return pd.concat(parts, ignore_index=True)

def season_weather(dist, today):
    """Observed daily weather for every district from 1 April, cached per season and topped up daily."""
    os.makedirs(LIVE_DATA, exist_ok=True)
    path = os.path.join(LIVE_DATA, f'weather_{today.year}.csv.gz')
    start = date(today.year, 4, 1)
    cache = pd.read_csv(path, parse_dates=['date']) if os.path.exists(path) else pd.DataFrame(columns=['outcode', 'date'])
    have = cache.date.max().date() if len(cache) else None
    need_from = start if have is None else have + timedelta(days=1)
    gap_end = today - timedelta(days=PAST_DAYS)   # the forecast API covers the last PAST_DAYS days
    if need_from < gap_end:
        log(f'bootstrapping {need_from} to {gap_end - timedelta(days=1)} from the historical-forecast archive')
        hist = pull_weather(dist, 'https://historical-forecast-api.open-meteo.com/v1/forecast',
                            f'start_date={need_from}&end_date={gap_end - timedelta(days=1)}', batch=10, pause=2.0)
        cache = pd.concat([cache, hist], ignore_index=True)
    log('pulling the last 14 days and the 8-day forecast')
    fc = pull_weather(dist, 'https://api.open-meteo.com/v1/forecast', f'past_days={PAST_DAYS}&forecast_days={FORECAST_DAYS}')
    fc['date'] = pd.to_datetime(fc.date)
    observed = fc[fc.date.dt.date < today]; forecast = fc[fc.date.dt.date >= today]
    cache = pd.concat([cache[~cache.date.isin(observed.date.unique())] if len(cache) else cache, observed], ignore_index=True)
    cache = cache.sort_values(['outcode', 'date']).reset_index(drop=True)
    cache.to_csv(path, index=False, float_format='%.2f')
    return cache, forecast

# ---- run: score -------------------------------------------------------------------------
def run():
    import joblib
    today = datetime.now(ZoneInfo('Europe/London')).date()
    year = today.year
    saved = joblib.load(os.path.join(MODEL_DIR, 'live_gbm_all.joblib')); model, cols = saved['model'], saved['columns']
    ctx = json.load(open(os.path.join(MODEL_DIR, 'live_context.json')))
    dist = pd.read_csv(os.path.join(PROC, 'districts.csv')).set_index('outcode')
    reports = fetch_reports(year)
    this_season = reports[reports.date.dt.year == year]
    log(f'{today}: {len(this_season)} confirmed reports so far in {year}, latest {this_season.date.max().date() if len(this_season) else None}')
    cache, forecast = season_weather(dist, today)
    daily = pd.concat([cache, forecast], ignore_index=True); daily['date'] = pd.to_datetime(daily.date)
    reps = dict(ord=reports.date.map(pd.Timestamp.toordinal).values, lat=reports.lat.values, lon=reports.lon.values,
                crop=reports.crop.values.astype(bool), outcode=reports.outcode.values)
    frames = []
    for oc, g in daily.groupby('outcode'):
        if oc not in dist.index: continue
        g = flags(g.set_index('date').sort_index()).reset_index()
        f = district_features(oc, g, dist, reps, float(ctx['prior_rate'].get(oc, 0.0)), True)
        if len(f): frames.append(f)
    feat = pd.concat(frames, ignore_index=True)
    feat = feat.merge(daily[['outcode', 'date', 'tmin', 'rh90h']], on=['outcode', 'date'], how='left')   # raw weather for the demo tracks
    feat['week'] = feat.doy // 7
    feat['clim_week'] = feat.week.map({int(k): v for k, v in ctx['clim_week'].items()}).fillna(ctx['base_rate'])
    feat['p'] = model.predict_proba(feat[cols].values.astype('float32'))[:, 1]
    feat['forecast'] = (feat.date.dt.date >= today).astype(int)
    t30, thut = ctx['threshold_30'], ctx['threshold_hutton_rate']
    # season so far: national series, scored where the outcome is already known
    nat = feat.groupby('date').agg(alert=('hutton_alert14', 'mean'), model30=('p', lambda s: float((s >= t30).mean())),
                                   model_hut=('p', lambda s: float((s >= thut).mean())), forecast=('forecast', 'max')).reset_index()
    rep_by_day = this_season.groupby(this_season.date.dt.floor('D')).size()
    nat['reports'] = nat.date.map(rep_by_day).fillna(0).astype(int)
    known = feat[feat.date.dt.date <= today - timedelta(days=7)]
    score = None
    if len(known) and 0 < known.y7.sum() < len(known):
        from sklearn.metrics import roc_auc_score
        from .evaluate import recall_at_rate, rate_for_recall
        yy = known.y7.values; hr = float(known.hutton_alert14.mean())
        hcatch = float(known.loc[known.hutton_alert14 == 1, 'y7'].sum() / yy.sum())
        score = dict(days=int(len(known)), positives=int(yy.sum()), through=str((today - timedelta(days=7))),
                     hutton_alert_share=hr, hutton_catch=hcatch, hutton_auc=float(roc_auc_score(yy, known.hutton_alert14.values)),
                     model_auc=float(roc_auc_score(yy, known.p.values)), model_catch_at_hutton_rate=float(recall_at_rate(yy, known.p.values, hr)),
                     model_rate_for_hutton_catch=float(rate_for_recall(yy, known.p.values, hcatch)))
    days = [today + timedelta(days=i) for i in range(FORECAST_DAYS)]
    ahead = feat[feat.date.dt.date.isin(days)]
    out_d = []
    for oc, g in ahead.groupby('outcode'):
        g = g.set_index(g.date.dt.date).reindex(days)
        r = lambda c, k: [None if pd.isna(v) else round(float(v), k) for v in g[c]]
        d = dist.loc[oc]
        out_d.append({'oc': oc, 'lat': round(float(d.lat), 3), 'lon': round(float(d.lon), 3), 'country': d.country if isinstance(d.country, str) else '',
                      'p': r('p', 4), 'alert': [None if pd.isna(v) else int(v) for v in g.hutton_alert14], 'hd14': r('huttonday_n14', 0),
                      'kern': r('kern_all', 2), 'near100': r('near100_28', 0), 'prior': round(float(ctx['prior_rate'].get(oc, 0.0)), 2)})
    tracks = {}
    for oc in DEMO:
        g = feat[feat.outcode == oc].set_index(feat[feat.outcode == oc].date.dt.date)
        if not len(g): continue
        idx = [date(year, 5, 1) + timedelta(days=i) for i in range(184)]
        g = g.reindex(idx)
        r = lambda c, k: [None if pd.isna(v) else round(float(v), k) for v in g[c]]
        tracks[oc] = {'p': r('p', 4), 'alert': [None if pd.isna(v) else int(v) for v in g.hutton_alert14], 'hd14': r('huttonday_n14', 0),
                      'kern': r('kern_all', 2), 'clim': r('clim_week', 4), 'tmin': [None if pd.isna(v) else int(round(v * 10)) for v in g.tmin],
                      'rh': r('rh90h', 0), 'forecast_from': int((today - date(year, 5, 1)).days),
                      'reports': [[int((d.date() - date(year, 5, 1)).days), int(c)] for d, c in zip(this_season[this_season.outcode == oc].date, this_season[this_season.outcode == oc].crop) if 0 <= (d.date() - date(year, 5, 1)).days < 184]}
    out = {'generated': datetime.now(ZoneInfo('Europe/London')).isoformat(timespec='minutes'), 'today': str(today), 'season': year,
           'days': [str(d) for d in days], 'fitted_seasons': ctx['fitted_seasons'],
           'thresholds': {'model_30': round(t30, 4), 'model_hutton_rate': round(thut, 4), 'base_rate': round(ctx['base_rate'], 4)},
           'districts': out_d,
           'reports': [{'oc': r.outcode, 'date': str(r.date.date()), 'lat': round(float(r.lat), 3), 'lon': round(float(r.lon), 3), 'crop': int(r.crop)} for r in this_season.itertuples()],
           'season_series': [{'date': str(r.date.date()), 'alert': round(float(r.alert), 3), 'model30': round(float(r.model30), 3), 'model_hut': round(float(r.model_hut), 3),
                              'reports': int(r.reports), 'forecast': int(r.forecast)} for r in nat.itertuples()],
           'score': score, 'tracks': tracks, 'names': {'DD8': 'Angus (Forfar)', 'IP12': 'Suffolk (Woodbridge)', 'CT3': 'Kent (Canterbury)', 'SA48': 'Ceredigion (Lampeter)', 'TF10': 'Shropshire (Newport)', 'TR12': 'Cornwall (TR12)', 'AB53': 'Aberdeenshire (AB53)'}}
    os.makedirs(LIVE_OUT, exist_ok=True)
    json.dump(out, open(os.path.join(LIVE_OUT, 'latest.json'), 'w'), separators=(',', ':'))
    tod = ahead[ahead.date.dt.date == today]
    log(f'today: Hutton alert on {tod.hutton_alert14.mean():.2f} of districts, model above its 30 % threshold on {(tod.p >= t30).mean():.2f}, '
        f'top districts: {", ".join(f"{r.outcode} {r.p:.2f}" for r in tod.nlargest(5, "p").itertuples())}')
    if score: log('season so far:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in score.items()})
    log(f'written live/latest.json ({os.path.getsize(os.path.join(LIVE_OUT, "latest.json")) // 1024} KB)')
    log('DONE run')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    {'fit': fit, 'run': run}[cmd](*sys.argv[2:])
