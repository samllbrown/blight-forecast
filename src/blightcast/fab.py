"""Clean the Fight Against Blight outbreak export into data/processed/outbreaks.csv.

Rules (all documented in README):
- keep status == confirmed (drops 10 negatives, 2 pending; negatives only recorded from 2025)
- keep host == potato (drops 49 tomato, 11 other)
- drop rows without a postcode district or coordinates
- drop reports dated 1 Jan - 15 Mar: placeholder dates (7 rows are exactly 1 Jan) or stored-tuber finds,
  outside any growing season
- crop = source in {Conventional crop, Organic crop, Trial}; the rest (volunteer, dump, garden,
  unknown) are kept with crop == 0 so the analysis can be run on either set
Districts: modal centroid per outcode (74 outcodes carry more than one coordinate in the export).
"""
import glob, json, os, collections
import pandas as pd
from . import RAW, PROC

CROP = {'Conventional crop', 'Organic crop', 'Trial'}

def load_raw():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, 'fab', '20*.json'))):
        for r in json.load(open(f)):
            r['file_year'] = int(os.path.basename(f)[:4]); rows.append(r)
    return pd.DataFrame(rows)

def main():
    df = load_raw()
    n0 = len(df)
    df = df[df.status.eq('confirmed') & df.host.eq('potato')]
    df = df.dropna(subset=['outcode', 'realLatitude', 'realLongitude'])
    df = df[df.outcode.ne('')]
    df['date'] = pd.to_datetime(df.dateSubmitted)
    df = df[~((df.date.dt.month == 1) | (df.date.dt.month == 2) | ((df.date.dt.month == 3) & (df.date.dt.day <= 15)))]
    df['year'] = df.date.dt.year
    df['crop'] = df.sourceName.isin(CROP).astype(int)
    cent = (df.groupby(['outcode', 'realLatitude', 'realLongitude']).size().reset_index(name='n')
              .sort_values(['outcode', 'n'], ascending=[True, False]).drop_duplicates('outcode')
              .rename(columns={'realLatitude': 'lat', 'realLongitude': 'lon'})[['outcode', 'lat', 'lon']])
    counts = df.groupby('outcode').size().rename('n_reports')
    crop_counts = df[df.crop == 1].groupby('outcode').size().rename('n_crop_reports')
    country = df.groupby('outcode').country.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else '')
    dist = cent.set_index('outcode').join([counts, crop_counts, country]).fillna({'n_crop_reports': 0}).reset_index()
    out = df[['outbreakId', 'outbreakCode', 'date', 'year', 'outcode', 'country', 'itlNuts', 'sourceName', 'crop',
              'severityName', 'reportedVarietyName', 'isPublic']].sort_values('date')
    os.makedirs(PROC, exist_ok=True)
    out.to_csv(os.path.join(PROC, 'outbreaks.csv'), index=False)
    dist.to_csv(os.path.join(PROC, 'districts.csv'), index=False)
    print(f'raw {n0} -> kept {len(out)} ({out.crop.sum()} crop) in {len(dist)} districts; '
          f'years {out.year.min()}-{out.year.max()}')
    print(out.groupby('year').agg(n=('crop', 'size'), crop=('crop', 'sum')).T.to_string())

if __name__ == '__main__':
    main()
