# Retesting the Hutton Criteria for potato blight

Late blight (*Phytophthora infestans*) is the disease British potato growers spray against most. The warning they get, through Fight Against Blight and BlightSpy, is the **Hutton Criteria** (Dancey, Skelsey and Cooke, 2017): two consecutive days with a minimum air temperature of at least 10 °C and at least six hours with relative humidity at or above 90 %. It replaced the **Smith Period** (Smith, 1956; eleven humid hours instead of six). This project asks:

1. **How good is the rule as a warning, scored against days when blight did not follow?** Every published evaluation (Dancey 2017; Skelsey 2021) scores only "was there an alert in the 28 days before a report", with no negative class. Nobody has evaluated the rule on the nine seasons since it went live.
2. **Can cheap terms the rule ignores (day of season, what has been reported nearby, cumulative humidity) give the same catch rate with fewer alert days, honestly, out of sample?**
3. **Where does the rule work and where does it not?** A district-level map of Britain.

Everything is reproducible from public data: the Fight Against Blight outbreak export (James Hutton Institute), ERA5-Land reanalysis via Open-Meteo, NOAA ISD hourly station observations, and Wikipedia-derived postcode district polygons.

Status: **in progress** (started 8 September 2026). Numbers below are interim until the project is marked complete.

## Data

**Outbreaks.** `https://blight.hutton.ac.uk/api/outbreaks?year=YYYY` returns every scout-reported outbreak with report date, postcode district and its centroid, source (conventional crop, organic, trial, volunteer, dump, garden), variety, severity and lab status. Raw JSON in `data/raw/fab/`. Cleaning (`blightcast/fab.py`): confirmed potato outbreaks only, drop January to mid-March placeholder dates. 3,347 raw rows become 3,259 reports (2,452 on crops) in 525 districts, 2006 to 2026.

**Weather.** Hourly 2 m temperature and relative humidity at each district centroid. Primary source under evaluation: ERA5-Land through the Open-Meteo archive API (`scripts/pull_weather.py`, gzipped raw in `data/raw/weather/`), free but quota-limited to roughly twenty districts a day. Station alternative: NOAA Integrated Surface Database hourly observations for UK synoptic stations, with RH derived from dew point. Daily aggregates and rule flags in `blightcast/weather.py`.

**Geography.** Postcode district polygons from `missinglink/uk-postcode-polygons` (Wikipedia KML export) in `data/raw/geo/`.

**Weather, station product (primary).** Meteostat bulk hourly files (`scripts/pull_meteostat.py`, 100 GB stations with RH, no registration) give daily minimum temperature and hours at RH >= 90 % at each station; `blightcast/stations.py` interpolates these to district centroids by inverse-distance-squared weighting over the four nearest reporting stations within 80 km (nearest station a median 22 km away), and applies the rules to the interpolated values, as Blightwatch did with Met Office stations. No lapse-rate correction.

## Method

`blightcast/dataset.py` builds a district-day panel for 1 May to 31 October each season. Features use only information available on the day: rule flags and humidity/temperature/rain windows up to day t; reports dated up to t-1 within 20, 40, 60 and 100 km and in the district itself; national and regional season-to-date counts; the district's report rate in prior seasons. The outcome is a report in that district within the next 7 days (also 14, and crop-only variants).

`blightcast/evaluate.py`:
- **Part A** reproduces the published metric: share of reports with a Hutton period in the prior 28 days, next to the share of season days carrying an alert.
- **Part B** scores with the negative class: AUC, recall at the Hutton alert rate, alert rate needed to match Hutton recall, and within-district-season AUC. Fitted models (logistic, gradient boosting) are trained on seasons before the test season only.

## Results

Station-interpolated weather, 525 districts, seasons 2006 to 2025 (Meteostat bulk files end in March 2026, so the 2026 season is not yet scored). Test seasons 2012 to 2025, 1.35 million district-days, 12,394 of them followed by a report in that district within seven days. Tables in `results/tables/`, figures in `results/figures/`.

### Part A: the published metric reproduces

| | This project | Published |
|---|---|---|
| Reports with a Hutton period in the prior 28 days | 95.2 % (95.6 % crops only) | 96.1 % (Skelsey 2021, 2012 to 2017) |
| Reports with a Smith period in the prior 28 days | 67.7 % | 81.8 % (Skelsey 2021) |
| Days under a Hutton alert, June to September | 73 % | 31 % alert days (Skelsey 2021); 10 % of pre-outbreak days (Dancey 2017) |

The catch rate matches. The alert-day share is higher here because "under alert" means a Hutton period ended in the previous 14 days, which is how the alert is used, whereas the published figures count the day a period is declared.

### Part B: with the negative class

Held out season by season (train on earlier seasons only). Outcome: a report in the district within the next seven days.

| Signal known on the day | AUC | Catch rate at the Hutton alert rate (61 % of days) | Alert days needed for the Hutton catch rate (84 %) | AUC within a district-season |
|---|---|---|---|---|
| Hutton alert as issued | 0.62 | 0.84 | 61 % | 0.60 |
| Hutton days in the last 14 days | 0.69 | 0.86 | 58 % | 0.68 |
| Smith periods in the last 28 days | 0.61 | 0.71 | 80 % | 0.60 |
| Week of year (prior seasons) | 0.70 | 0.90 | 51 % | 0.72 |
| Reports within 100 km in the last 28 days | 0.76 | 0.89 | 47 % | 0.69 |
| District rate in prior seasons | 0.72 | 0.83 | 66 % | 0.50 |
| Model: weather only (gradient boosting) | 0.69 | 0.88 | 56 % | 0.68 |
| Model: calendar and reports | 0.80 | 0.93 | 42 % | 0.72 |
| Model: calendar, reports, district prior | 0.85 | 0.95 | 31 % | 0.74 |
| Model: everything (adds weather) | 0.86 | 0.96 | 30 % | 0.75 |

Read across the first row: the alert is on for 61 % of season days and catches 84 % of outbreak-weeks, which is a little better than chance (AUC 0.62). A calendar does better. Nearby reports do better still. The full model matches the Hutton catch rate using 30 % of the alert days, and weather adds one point of AUC on top of calendar and reports.

### Second run: richer features, lagged and neighbourhood outcomes

Changes: humid hours split by temperature band, runs of humid days, days since the last Hutton period, wind, night RH, thermal time since April, a decaying inoculum kernel over reports (30 km, 10 day scales, crop and non-crop separately), district latitude, longitude and elevation; training negatives subsampled to a quarter with compensating weights. Four outcomes: a report in the district within 7 days (`y7`), in the district 7 to 21 days ahead (`y7_lag`, allowing for report lag), within 25 km within 7 days (`yn25_7`), within 25 km 7 to 21 days ahead (`yn25_lag`).

| AUC, season-forward | y7 | y7_lag | yn25_7 | yn25_lag |
|---|---|---|---|---|
| Hutton alert as issued | 0.62 | 0.60 | 0.60 | 0.59 |
| Weather, basic features | 0.68 | 0.64 | 0.67 | 0.64 |
| Weather, rich features | 0.75 | 0.74 | 0.73 | 0.72 |
| Inoculum kernel alone | 0.79 | 0.71 | 0.73 | 0.72 |
| Calendar, reports, district prior | 0.85 | 0.83 | 0.83 | 0.80 |
| Everything | 0.86 | 0.84 | 0.84 | 0.81 |
| Alert days for the Hutton catch rate, everything | 30 % | 28 % | 33 % | 35 % |

Richer weather features are worth six to ten AUC points to the weather-only model on every outcome, and the inoculum kernel beats every fixed-window report count. The full model gains one point or less from weather on top of calendar, reports and place, on every outcome. Neither the lag window nor the 25 km neighbourhood raised weather's contribution, so the overlap between weather and nearby reports is not an artefact of the district unit or the report delay.

### Where the rule's signal comes from

By week of year (`by_week.stations.y7.csv`): in May and early June the alert is on for 10 to 40 % of days and its within-week AUC is 0.64 to 0.67. From late June to early September it is on for 70 to 88 % of days and its within-week AUC sits between 0.52 and 0.59. The rule discriminates when blight is rare and stops discriminating once the season is under way.

By region: alert share Wales 75 %, England 61 %, Scotland 54 %; Hutton AUC Scotland 0.66, England 0.60, Wales 0.59. By season: Hutton catch rate ranged from 0.58 (2022) and 0.69 (2024) to 0.96 (2012) and 0.98 (2021).

### Lag

Detrended within season, the correlation between the national share of districts under a Hutton period and the national daily report count peaks at a lag of 8 days, but weakly (r 0.07; raw r 0.20). Reports sit a median 3 days after the last Hutton period, against 5 days for random summer days in the same districts (75th percentile 9 versus 15 days).

### Weather source check

For the nine districts with both sources (May to September): ERA5-Land minimum temperature runs 0.6 °C warmer than the station interpolation and gives 5.3 hours at RH >= 90 % per day against 6.9 from stations; Hutton-day flags agree on 83 % of days and the period share is 17.5 % against 19.7 %. The coastal Kent and Suffolk districts differ most. Stations are used throughout.

## Limitations

- Dates are report dates, one to three weeks after infection; the lag is not recorded.
- Negatives are contaminated: unscouted or sprayed crops look like no blight. Scouts also receive Hutton alerts, so detection is partly alert-induced.
- Locations are postcode district centroids, and reanalysis under-counts humid night hours relative to stations.
- The nearby-report term is a proxy for how bad the season is in a region; a permutation test found no space-time clustering beyond that.
