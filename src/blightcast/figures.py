"""README figures (matplotlib PNG) from results tables.
  fig_curves.png      recall vs alert-rate curves with the Hutton operating point
  fig_by_week.png     weekly report rate, Hutton alert share, and AUC by week
  fig_map.png         district map: Hutton within-district AUC (or alert share) and report counts
  fig_lag.png         cross-correlation by lag
Usage: python -m blightcast.figures [model] [y]
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from . import RAW, RES, PROC

def curves(model, y):
    c = pd.read_csv(os.path.join(RES, 'tables', f'curves.{model}.{y}.csv'))
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, g in c.groupby('model', sort=False):
        g2 = g.dropna()
        if name == 'Hutton alert':
            pt = g[g.recall.isna()].alert_rate.iloc[0]
            gr = pd.read_csv(os.path.join(RES, 'tables', f'groups.{model}.{y}.csv')); r = gr[gr.group == 'all'].iloc[0]
            ax.plot(r.hutton_alert_rate, r.hutton_recall, 'o', ms=10, color='black', label='Hutton alert as issued', zorder=5)
            continue
        ax.plot(g2.alert_rate, g2.recall, label=name)
    ax.plot([0, 1], [0, 1], ':', color='grey', label='no skill')
    ax.set_xlabel('share of district-days under alert'); ax.set_ylabel('share of outbreak-weeks caught')
    ax.set_title('Catch rate against alert days, seasons 2012 to 2025, held out season by season')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(RES, 'figures', 'fig_curves.png'), dpi=150)

def by_week(model, y):
    w = pd.read_csv(os.path.join(RES, 'tables', f'by_week.{model}.{y}.csv'))
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax = axes[0]; ax.plot(w.doy_start, w.rate, label='share of district-days followed by a report within 7 d'); ax.plot(w.doy_start, w.hutton_alert_share, label='share of district-days under Hutton alert'); ax.legend(fontsize=8); ax.set_ylabel('share')
    ax = axes[1]; ax.plot(w.doy_start, w.hutton_auc, label='Hutton alert'); ax.plot(w.doy_start, w.huttonday14_auc, label='Hutton days in 14 d'); ax.plot(w.doy_start, w.near100_auc, label='reports within 100 km'); ax.plot(w.doy_start, w.best_auc, label='model: all'); ax.axhline(0.5, color='grey', ls=':'); ax.set_ylabel('AUC within the week'); ax.set_xlabel('day of year'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'figures', 'fig_by_week.png'), dpi=150)

def lag(model):
    l = pd.read_csv(os.path.join(RES, 'tables', f'lag_xcorr.{model}.csv'))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(l.lag_days, l.r, label='detrended within season'); ax.plot(l.lag_days, l.r_raw, label='raw', alpha=0.5)
    ax.axvline(0, color='grey', ls=':'); ax.set_xlabel('days from Hutton period to report'); ax.set_ylabel('correlation'); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(RES, 'figures', 'fig_lag.png'), dpi=150)

def district_map(model, y):
    pr = pd.read_csv(os.path.join(RES, 'tables', f'preds.{model}.{y}.csv.gz'))
    from sklearn.metrics import roc_auc_score
    rows = []
    for oc, g in pr.groupby('outcode'):
        if 0 < g[y].sum() < len(g) and g[y].sum() >= 10:
            rows.append(dict(outcode=oc, n=int(g[y].sum()), hutton_auc=roc_auc_score(g[y], g.hutton_alert14), best_auc=roc_auc_score(g[y], g.p_gbm_all),
                             alert_share=g.hutton_alert14.mean()))
    d = pd.DataFrame(rows).set_index('outcode')
    polys = {}
    for f in glob.glob(os.path.join(RAW, 'geo', '*.geojson')):
        for ft in json.load(open(f))['features']:
            polys[ft['properties']['name']] = ft['geometry']
    def patches(geom):
        if not geom: return []
        if geom['type'] == 'Polygon': return [Polygon(np.array(r), closed=True) for r in geom['coordinates'][:1]]
        if geom['type'] == 'MultiPolygon': return [Polygon(np.array(p[0]), closed=True) for p in geom['coordinates']]
        if geom['type'] == 'GeometryCollection': return [q for g in geom['geometries'] for q in patches(g)]
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, col, title, cmap, vmin, vmax in [(axes[0], 'n', 'reports 2012 to 2025', 'Purples', 0, None),
                                            (axes[1], 'hutton_auc', 'Hutton alert AUC in district', 'RdYlGn', 0.3, 0.8),
                                            (axes[2], 'best_auc', 'model AUC in district', 'RdYlGn', 0.3, 0.9)]:
        bg = [p for oc, gm in polys.items() for p in patches(gm)]
        ax.add_collection(PatchCollection(bg, facecolor='#eeeeee', edgecolor='none'))
        ps, vals = [], []
        for oc, r in d.iterrows():
            if oc in polys:
                for p in patches(polys[oc]): ps.append(p); vals.append(r[col])
        pc = PatchCollection(ps, cmap=cmap, edgecolor='none'); pc.set_array(np.array(vals)); pc.set_clim(vmin, vmax if vmax else np.percentile(vals, 95))
        ax.add_collection(pc); fig.colorbar(pc, ax=ax, shrink=0.6)
        ax.set_xlim(-8, 2); ax.set_ylim(49.8, 59); ax.set_aspect(1.6); ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'figures', 'fig_map.png'), dpi=130)
    d.to_csv(os.path.join(RES, 'tables', f'district_scores.{model}.{y}.csv'))

def main(model='stations', y='y7'):
    os.makedirs(os.path.join(RES, 'figures'), exist_ok=True)
    curves(model, y); by_week(model, y); lag(model); district_map(model, y); print('figures written')

if __name__ == '__main__':
    main(*sys.argv[1:])
