"""Score the Hutton Criteria and rivals on the district-day panel.

Part A reproduces the published metric (Dancey 2017; Skelsey 2021): for each report, was a
Hutton period issued in the 28 days before the report date; alongside the share of season days
that carry an alert. No negative class.

Part B adds the negative class. Every district-day in the season window is a case; the outcome is
a report (in the district, or within 25 km) in a window after the day. Metrics: AUC; recall at the
Hutton alert rate; alert rate needed for the Hutton recall; within-district-season AUC. Models are
fitted season-forward: train on seasons before y, test on y. Training negatives are subsampled
(NEG_FRAC) with compensating weights; test sets are complete.

Usage: python -m blightcast.evaluate <model> <outcome> [<outcome> ...]
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from . import PROC, RES
import time

def progress(**ev):
    """Append one JSON event to results/progress.jsonl for the live dashboard."""
    ev['ts'] = time.time()
    with open(os.path.join(RES, 'progress.jsonl'), 'a') as f: f.write(json.dumps(ev) + '\n')

FIRST_TEST = 2012
NEG_FRAC = 0.25
WEATHER_BASIC = ['hutton_alert14', 'hutton_n28', 'hutton_n14', 'huttonday_n14', 'smith_n28', 'rh90h_7', 'rh90h_14',
                 'wet10h_14', 'tmin_7', 'tmean_14', 'rain_7', 'rain_28']
WEATHER_RICH = WEATHER_BASIC + ['huttonday_n28', 'hutton_run', 'days_since_hp', 'rh90h_28', 'rh85h_14', 'rh95h_14', 'humid_run',
                                'tmax_7', 'raindays_14', 'w_t7_14', 'w_t12_14', 'w_t16_14', 'w_t22_14', 'w_t12_22_7', 'wspd_7',
                                'rh_night_7', 'gdd_apr']
CALENDAR = ['doy', 'clim_week']
REPORTS = ['near20_14', 'near40_14', 'near40_28', 'near100_28', 'own_21', 'own_60', 'nat_ytd', 'reg100_ytd',
           'kern_all', 'kern_crop', 'kern_noncrop']
PRIOR = ['prior_rate']
PLACE = ['lat', 'lon', 'elev']
SETS = {
    'weather_basic': WEATHER_BASIC,
    'weather': WEATHER_RICH,
    'calendar+weather': CALENDAR + WEATHER_RICH,
    'calendar+reports': CALENDAR + REPORTS,
    'calendar+reports+prior': CALENDAR + REPORTS + PRIOR + PLACE,
    'all': CALENDAR + WEATHER_RICH + REPORTS + PRIOR + PLACE,
}
MODELS = {'gbm': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                        min_samples_leaf=100, l2_regularization=1.0),
          'logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))}
LOGISTIC_SETS = ('weather', 'all')

def recall_at_rate(y, s, rate):
    k = int(round(rate * len(s)))
    if k <= 0: return 0.0
    thr = np.sort(s)[::-1][k - 1]
    alerted = s > thr
    need = k - alerted.sum(); ties = np.where(s == thr)[0]
    if need > 0 and len(ties): alerted[ties[:need]] = True
    return y[alerted].sum() / max(1, y.sum())

def rate_for_recall(y, s, target):
    order = np.argsort(-s, kind='stable'); cum = np.cumsum(y[order]) / max(1, y.sum())
    k = np.searchsorted(cum, target, side='left') + 1
    return min(1.0, k / len(s))

def within_auc(df, score, y):
    vals, w = [], []
    for _, g in df.groupby(['outcode', 'year']):
        yy = g[y].values
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, g[score].values)); w.append(yy.sum())
    return float(np.average(vals, weights=w)) if vals else np.nan, len(vals)

def part_a(panel, ob):
    ob = ob[ob.date.dt.month.between(4, 10)]
    m = ob.merge(panel[['outcode', 'date', 'hutton_n28', 'smith_n28']], on=['outcode', 'date'], how='left').dropna(subset=['hutton_n28'])
    s = panel[panel.date.dt.month.between(6, 9)]
    return {'reports_scored': len(m), 'hutton_any_in_28d': float((m.hutton_n28 > 0).mean()),
            'hutton_any_in_28d_crop': float((m[m.crop == 1].hutton_n28 > 0).mean()), 'smith_any_in_28d': float((m.smith_n28 > 0).mean()),
            'alert_share_days_JJAS': float((s.hutton_n14 > 0).mean()), 'alert_share_days_MJJASO': float(panel.hutton_alert14.mean())}

def subsample(panel, mask, y, seed=0):
    rng = np.random.default_rng(seed)
    yv = panel[y].values
    keep = mask & ((yv == 1) | (rng.random(len(panel)) < NEG_FRAC))
    w = np.where(yv[keep] == 1, 1.0, 1.0 / NEG_FRAC)
    return panel.loc[keep], w

def part_b(panel, y):
    rows = []; saved = {}
    test = panel[panel.year >= FIRST_TEST].reset_index(drop=True)
    clim = 'clim_week_n25' if y.startswith('yn25') else 'clim_week'
    singles = {'Hutton alert (period in last 14 d)': 'hutton_alert14', 'Hutton periods in last 28 d': 'hutton_n28',
               'Hutton days in last 14 d': 'huttonday_n14', 'Smith periods in last 28 d': 'smith_n28',
               'RH>=90 hours last 14 d': 'rh90h_14', 'Reports within 40 km last 14 d': 'near40_14',
               'Reports within 100 km last 28 d': 'near100_28', 'Inoculum kernel (all reports)': 'kern_all',
               'Own district reports last 21 d': 'own_21', 'National reports season to date': 'nat_ytd',
               'Prior-season district rate': 'prior_rate', 'Week-of-year climatology (prior seasons)': clim}
    hut_rate = float(test.hutton_alert14.mean()); hut_recall = float(test.loc[test.hutton_alert14 == 1, y].sum() / test[y].sum())
    for name, col in singles.items():
        s, yy = test[col].values.astype(float), test[y].values
        wa, n = within_auc(test, col, y)
        rows.append(dict(model=name, kind='single', auc=roc_auc_score(yy, s), recall_at_hutton_rate=recall_at_rate(yy, s, hut_rate),
                         rate_for_hutton_recall=rate_for_recall(yy, s, hut_recall), within_auc=wa, within_n=n))
    for setname, cols in SETS.items():
        cols = [c.replace('clim_week', clim) if c == 'clim_week' else c for c in cols]
        cols = [c for c in cols if c in panel.columns]
        for mname, mk in MODELS.items():
            if mname == 'logistic' and setname not in LOGISTIC_SETS: continue
            preds = np.full(len(test), np.nan)
            seasons = sorted(test.year.unique())
            progress(event='start', outcome=y, set=setname, model=mname, n_seasons=len(seasons))
            for yr in seasons:
                mask = (panel.year < yr).values
                if panel.loc[mask, y].sum() < 20: continue
                trs, w = subsample(panel, mask, y)
                m = mk()
                if mname == 'gbm': m.fit(trs[cols].values, trs[y].values, sample_weight=w)
                else: m.fit(trs[cols].values, trs[y].values, logisticregression__sample_weight=w)
                sel = (test.year == yr).values
                preds[sel] = m.predict_proba(test.loc[sel, cols].values)[:, 1]
                ys = test.loc[sel, y].values
                progress(event='season', outcome=y, set=setname, model=mname, season=int(yr),
                         auc=float(roc_auc_score(ys, preds[sel])) if 0 < ys.sum() < len(ys) else None)
            ok = ~np.isnan(preds); t2 = test[ok].assign(_p=preds[ok])
            saved[f'p_{mname}_{setname.replace("+", "_")}'] = preds
            wa, n = within_auc(t2, '_p', y)
            per_season = {int(yr): float(roc_auc_score(g[y], g._p)) if 0 < g[y].sum() < len(g) else None for yr, g in t2.groupby('year')}
            rows.append(dict(model=f'{mname}: {setname}', kind='fitted', auc=roc_auc_score(t2[y], t2._p),
                             recall_at_hutton_rate=recall_at_rate(t2[y].values, t2._p.values, hut_rate),
                             rate_for_hutton_recall=rate_for_recall(t2[y].values, t2._p.values, hut_recall),
                             within_auc=wa, within_n=n, per_season=json.dumps(per_season)))
            print(f'  {y} {mname}: {setname} auc {rows[-1]["auc"]:.3f}', flush=True)
            progress(event='done', outcome=y, set=setname, model=mname, auc=float(rows[-1]['auc']),
                     recall_at_hutton_rate=float(rows[-1]['recall_at_hutton_rate']), rate_for_hutton_recall=float(rows[-1]['rate_for_hutton_recall']),
                     within_auc=float(rows[-1]['within_auc']))
    res = pd.DataFrame(rows)
    res.attrs = dict(hutton_rate=hut_rate, hutton_recall=hut_recall, n=len(test), positives=int(test[y].sum()))
    keep = ['outcode', 'region', 'date', 'year', 'doy', y, 'hutton_alert14', 'hutton_n28', 'huttonday_n14', clim, 'near100_28',
            'near40_14', 'kern_all', 'prior_rate']
    preds_out = test[keep].rename(columns={clim: 'clim_week'}).copy()
    for k, v in saved.items(): preds_out[k] = v
    res.attrs['preds'] = preds_out
    return res

def main(model='era5_land', *outcomes):
    outcomes = outcomes or ('y7',)
    need = set(sum(SETS.values(), [])) | {'clim_week_n25', 'outcode', 'region', 'date', 'year', 'week', 'doy', 'hutton_alert14',
                                          'hutton_n28', 'huttonday_n14', 'smith_n28', 'near100_28', 'near40_14', 'kern_all',
                                          'prior_rate', 'own_21', 'nat_ytd', 'rh90h_14'} | set(outcomes)
    head = pd.read_csv(os.path.join(PROC, f'panel.{model}.csv.gz'), nrows=1).columns
    usecols = [c for c in head if c in need]
    dtypes = {c: 'float32' for c in usecols if c not in ('outcode', 'region', 'date', 'year', 'week', 'doy') and not c.startswith('y')}
    panel = pd.read_csv(os.path.join(PROC, f'panel.{model}.csv.gz'), parse_dates=['date'], usecols=usecols, dtype=dtypes)
    for c in [c for c in usecols if c.startswith('y') and c != 'year']: panel[c] = panel[c].astype('int8')
    print(f'panel {len(panel)} rows x {len(usecols)} cols, {panel.memory_usage(deep=True).sum() / 1e9:.2f} GB', flush=True)
    ob = pd.read_csv(os.path.join(PROC, 'outbreaks.csv'), parse_dates=['date'])
    ob = ob[ob.outcode.isin(panel.outcode.unique())]
    os.makedirs(os.path.join(RES, 'tables'), exist_ok=True)
    progress(event='run', model_source=model, outcomes=list(outcomes), rows=len(panel))
    a = part_a(panel, ob); print('Part A (published metric):', json.dumps(a, indent=1), flush=True)
    json.dump(a, open(os.path.join(RES, 'tables', f'part_a.{model}.json'), 'w'), indent=1)
    pd.set_option('display.width', 220)
    for y in outcomes:
        progress(event='outcome', outcome=y, positives=int(panel.loc[panel.year >= FIRST_TEST, y].sum()))
        b = part_b(panel, y)
        b.to_csv(os.path.join(RES, 'tables', f'part_b.{model}.{y}.csv'), index=False)
        b.attrs['preds'].to_csv(os.path.join(RES, 'tables', f'preds.{model}.{y}.csv.gz'), index=False, float_format='%.4f')
        print(f"\nPart B [{y}]: {b.attrs['n']} test district-days, {b.attrs['positives']} positives, Hutton alert on "
              f"{b.attrs['hutton_rate']:.3f} of days with recall {b.attrs['hutton_recall']:.3f}")
        print(b.drop(columns=['per_season'], errors='ignore').to_string(index=False, float_format=lambda v: f'{v:.3f}'), flush=True)

if __name__ == '__main__':
    main(*sys.argv[1:])
