"""Secondary analyses on the saved predictions and panel:
  - per-region and per-season scores for the Hutton alert and the best model
  - Hutton skill within week-of-year strata (how much of the rule's signal is the calendar)
  - lag: cross-correlation between the national share of districts under a Hutton period and the
    national daily report count, by lag in days; and days from last Hutton period start to report
  - recall vs alert-rate curves (saved as a table for figures)
Usage: python -m blightcast.analysis [model] [y]
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from . import PROC, RES
from .evaluate import recall_at_rate, rate_for_recall, within_auc

def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan

def main(model='stations', y='y7'):
    pr = pd.read_csv(os.path.join(RES, 'tables', f'preds.{model}.{y}.csv.gz'), parse_dates=['date'])
    best = 'p_gbm_all'
    hut_rate = pr.hutton_alert14.mean(); hut_recall = pr.loc[pr.hutton_alert14 == 1, y].sum() / pr[y].sum()
    rows = []
    for key, g in [('all', pr)] + list(pr.groupby('region')) + list(pr.groupby('year')):
        yy, hr = g[y].values, g.hutton_alert14.mean()
        rows.append(dict(group=str(key), n=len(g), positives=int(yy.sum()), hutton_alert_rate=hr,
                         hutton_recall=g.loc[g.hutton_alert14 == 1, y].sum() / max(1, yy.sum()),
                         hutton_auc=safe_auc(yy, g.hutton_alert14.values), huttonday14_auc=safe_auc(yy, g.huttonday_n14.values),
                         clim_auc=safe_auc(yy, g.clim_week.values), near100_auc=safe_auc(yy, g.near100_28.values),
                         best_auc=safe_auc(yy, g[best].values),
                         best_recall_at_hutton_rate=recall_at_rate(yy, g[best].values, hr),
                         best_rate_for_hutton_recall=rate_for_recall(yy, g[best].values, g.loc[g.hutton_alert14 == 1, y].sum() / max(1, yy.sum()))))
    groups = pd.DataFrame(rows); groups.to_csv(os.path.join(RES, 'tables', f'groups.{model}.{y}.csv'), index=False)
    # within week-of-year strata: AUC of Hutton signals pooled over districts and years inside each week
    wk = []
    for w, g in pr.groupby(pr.doy // 7):
        wk.append(dict(week=int(w), doy_start=int(w * 7), n=len(g), positives=int(g[y].sum()), rate=g[y].mean(),
                       hutton_alert_share=g.hutton_alert14.mean(), hutton_auc=safe_auc(g[y].values, g.hutton_alert14.values),
                       huttonday14_auc=safe_auc(g[y].values, g.huttonday_n14.values), near100_auc=safe_auc(g[y].values, g.near100_28.values),
                       best_auc=safe_auc(g[y].values, g[best].values)))
    pd.DataFrame(wk).to_csv(os.path.join(RES, 'tables', f'by_week.{model}.{y}.csv'), index=False)
    # recall vs alert-rate curves
    curves = []
    for name, col in [('Hutton alert', 'hutton_alert14'), ('Hutton days in last 14 d', 'huttonday_n14'),
                      ('Week-of-year climatology', 'clim_week'), ('Reports within 100 km, 28 d', 'near100_28'),
                      ('Model: calendar + reports', 'p_gbm_calendar_reports'), ('Model: weather', 'p_gbm_weather'), ('Model: all', best)]:
        s = pr[col].values.astype(float); yy = pr[y].values
        order = np.argsort(-s, kind='stable'); cum = np.cumsum(yy[order]) / yy.sum()
        for r in np.linspace(0.02, 1.0, 50):
            k = int(r * len(s)); curves.append(dict(model=name, alert_rate=r, recall=cum[k - 1]))
        curves.append(dict(model=name, alert_rate=float((s > 0).mean()) if col == 'hutton_alert14' else np.nan, recall=np.nan))
    pd.DataFrame(curves).to_csv(os.path.join(RES, 'tables', f'curves.{model}.{y}.csv'), index=False)
    # lag analysis on the panel
    panel = pd.read_csv(os.path.join(PROC, f'panel.{model}.csv.gz'), parse_dates=['date'], usecols=['outcode', 'date', 'year', 'y0', 'hutton_n14', 'hutton_alert14'])
    daily = pd.read_csv(os.path.join(PROC, f'daily.{model}.csv.gz'), parse_dates=['date'], usecols=['outcode', 'date', 'hutton_period'])
    ob = pd.read_csv(os.path.join(PROC, 'outbreaks.csv'), parse_dates=['date'])
    nat = daily[daily.date.dt.month.between(5, 10)].groupby('date').hutton_period.mean().rename('hp_share')
    rep = ob.groupby('date').size().rename('reports')
    ts = pd.concat([nat, rep], axis=1).fillna({'reports': 0}).dropna(subset=['hp_share'])
    ts = ts[ts.index.month.isin([6, 7, 8, 9])]
    # within-year detrended cross-correlation: subtract each year's weekly-smoothed mean to remove the seasonal curve
    ts['year'] = ts.index.year
    for c in ('hp_share', 'reports'):
        ts[c + '_a'] = ts[c] - ts.groupby('year')[c].transform(lambda s: s.rolling(29, center=True, min_periods=7).mean())
    lags = []
    for L in range(-7, 36):
        x = ts.hp_share_a.shift(L); lags.append(dict(lag_days=L, r=float(ts.reports_a.corr(x)), r_raw=float(ts.reports.corr(ts.hp_share.shift(L)))))
    lagdf = pd.DataFrame(lags); lagdf.to_csv(os.path.join(RES, 'tables', f'lag_xcorr.{model}.csv'), index=False)
    # days since the most recent Hutton period for each report, vs for random days in the same district-season
    d2 = daily.sort_values(['outcode', 'date'])
    d2['last_hp'] = d2.date.where(d2.hutton_period == 1)
    d2['last_hp'] = d2.groupby('outcode').last_hp.ffill()
    d2['since_hp'] = (d2.date - d2.last_hp).dt.days
    m = ob.merge(d2[['outcode', 'date', 'since_hp']], on=['outcode', 'date'], how='left')
    rnd = d2[d2.date.dt.month.between(6, 9) & d2.outcode.isin(ob.outcode.unique())].sample(50000, random_state=1)
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    since = pd.DataFrame({'reports': m.since_hp.quantile(q), 'random_summer_days': rnd.since_hp.quantile(q)})
    since.to_csv(os.path.join(RES, 'tables', f'since_hutton.{model}.csv'))
    print(groups.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nbest lag (detrended):', lagdf.loc[lagdf.r.idxmax()].to_dict())
    print('\ndays since last Hutton period, reports vs random summer days:\n', since.to_string())
    print('\nby week:\n', pd.DataFrame(wk).to_string(index=False, float_format=lambda v: f'{v:.3f}'))

if __name__ == '__main__':
    main(*sys.argv[1:])
