"""How the Hutton alert's share of days and catch rate depend on how long an alert is held.

"Under alert" in the panel means a Hutton period ended in the previous 14 days. The published
figures (Skelsey 2021: 31 % alert days) count only the day a period is declared. This scores the
same rule with the alert held for 1 to 28 days, on the test seasons, so the post can show the
choice and its consequences. The published metric (a period in the 28 days before a report) does
not depend on the hold and is printed for reference.

Usage: python -m blightcast.holdwindow [model] -> results/tables/hold_window.<model>.y7.csv
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from . import PROC, RES
from .evaluate import FIRST_TEST

HOLDS = [1, 2, 3, 5, 7, 10, 14, 21, 28]

def main(model='stations'):
    daily = pd.read_csv(os.path.join(PROC, f'daily.{model}.csv.gz'), parse_dates=['date'], usecols=['outcode', 'date', 'tmin', 'hutton_period'])
    ob = pd.read_csv(os.path.join(PROC, 'outbreaks.csv'), parse_dates=['date'])
    ob['ord'] = ob.date.map(pd.Timestamp.toordinal)
    acc = {h: dict(days=0, on=0, pos=0, caught=0, y=[], s=[]) for h in HOLDS}
    warned = 0; nrep = 0
    for oc, g in daily.groupby('outcode'):
        g = g.sort_values('date')
        full = pd.DataFrame({'date': pd.date_range(g.date.min(), g.date.max())}).merge(g, on='date', how='left')
        hp = full.hutton_period.fillna(0).values.astype(int)
        ords = full.date.map(pd.Timestamp.toordinal).values
        r = np.sort(ob.loc[ob.outcode == oc, 'ord'].values)
        y7 = (np.searchsorted(r, ords + 7, side='right') - np.searchsorted(r, ords, side='right') > 0).astype(int)
        keep = (full.date.dt.month.between(5, 10) & (full.date.dt.year >= FIRST_TEST) & full.tmin.notna()).values
        hp_s = pd.Series(hp)
        for h in HOLDS:
            on = hp_s.rolling(h, min_periods=1).sum().gt(0).values.astype(int)[keep]
            a = acc[h]; yy = y7[keep]
            a['days'] += len(on); a['on'] += int(on.sum()); a['pos'] += int(yy.sum()); a['caught'] += int((on & yy).sum())
            a['y'].append(yy.astype(np.int8)); a['s'].append(on.astype(np.int8))
        # published metric for reports in test seasons
        n28 = hp_s.rolling(28, min_periods=1).sum().values
        idx = {o: i for i, o in enumerate(ords)}
        for o in r:
            d = pd.Timestamp.fromordinal(int(o))
            if d.year < FIRST_TEST or not 4 <= d.month <= 10 or o not in idx: continue
            nrep += 1; warned += int(n28[idx[o]] > 0)
    rows = []
    for h in HOLDS:
        a = acc[h]; y = np.concatenate(a['y']); s = np.concatenate(a['s'])
        rows.append(dict(hold_days=h, alert_share=a['on'] / a['days'], catch_rate=a['caught'] / a['pos'], auc=float(roc_auc_score(y, s)),
                         published_metric=warned / nrep, days=a['days'], positives=a['pos']))
        print(f"hold {h:2d} d: alert on {rows[-1]['alert_share']:.3f} of days, catches {rows[-1]['catch_rate']:.3f}, auc {rows[-1]['auc']:.3f}", flush=True)
    out = pd.DataFrame(rows); os.makedirs(os.path.join(RES, 'tables'), exist_ok=True)
    out.to_csv(os.path.join(RES, 'tables', f'hold_window.{model}.y7.csv'), index=False)
    print(f'published metric (independent of hold): {warned / nrep:.3f} of {nrep} reports')
    print('DONE holdwindow')

if __name__ == '__main__':
    main(*sys.argv[1:])
