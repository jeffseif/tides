# tides

A library for forecasting tide levels

## Why did you make this library?

There honestly isn't a huge need for this library.  However, I wanted to learn about different timeseries forecasting approaches and find that experimentation is often an effective way to do that!  I also have been curious about how different approaches perform on periodic data like tidal levels.

## How do you use it?

There is a `forecast` make target backed by a nice entrypoint which fetches historical observations for a buoy, fits a model to that data, and predicts tide levels for a day from that model.  Below, I call the entrypoint explicitly with the default arguments for reproducibility:

```shell
uv run python -m tides --model-name=FittedHarmonicModel --number-of-training-months=48 --station-number=9414290 --verbose forecast --day=2026-03-02 ;
INFO:tides.model:Forecasting (MTL) tide levels for station_number=9414290 for 2026-03-02
INFO:tides.model:Fetching historical (MTL) tide levels for station_number=9414290 between 2022-03 and 2026-02 (inclusive) for training
INFO:tides.model:Received 350_640 data points with 14_179 NaN's
INFO:tides.model:Fitting FittedHarmonicModel model with parameters={'learning_rate': 3}
INFO:tides.model:Found historical mean[MTL] of 0.0786m and harmonics:
M2   = 0.539m @ 1.932d -0.788rad
K1   = 0.393m @ 1.003d +2.470rad
O1   = 0.255m @ 0.930d -0.938rad
S2   = 0.135m @ 2.000d +2.500rad
N2   = 0.116m @ 1.896d +1.493rad
P1   = 0.104m @ 0.997d +2.200rad
K2   = 0.057m @ 2.005d -0.281rad
Q1   = 0.046m @ 0.893d +1.013rad
MK3  = 0.024m @ 2.935d +0.999rad
J1   = 0.022m @ 1.039d +0.300rad
M4   = 0.022m @ 3.865d -2.691rad
NU2  = 0.020m @ 1.901d +0.736rad
2MK3 = 0.019m @ 2.862d -2.147rad
OO1  = 0.019m @ 1.076d +2.274rad
S1   = 0.018m @ 1.000d +2.375rad
L2   = 0.018m @ 1.969d +0.289rad
M1   = 0.018m @ 0.966d -1.810rad
2N2  = 0.017m @ 1.860d -2.495rad
SSA  = 0.016m @ 0.005d -0.863rad
T2   = 0.015m @ 1.997d +2.212rad
MF   = 0.013m @ 0.073d +3.141rad
MS4  = 0.012m @ 3.932d +0.588rad
LAM2 = 0.010m @ 1.964d +1.477rad
MN4  = 0.010m @ 3.828d -0.415rad
RHO  = 0.010m @ 0.898d +0.179rad
MSF  = 0.006m @ 0.068d -0.000rad
2SM2 = 0.006m @ 2.068d +2.638rad
MU2  = 0.005m @ 1.865d -2.127rad
SA   = 0.005m @ 0.003d -0.064rad
R2   = 0.005m @ 2.003d -2.715rad
M3   = 0.004m @ 2.898d -2.021rad
M6   = 0.003m @ 5.797d +2.363rad
2Q1  = 0.003m @ 0.857d +2.914rad
S6   = 0.000m @ 6.000d -1.951rad
S4   = 0.000m @ 4.000d -1.510rad
M8   = 0.000m @ 7.729d -2.391rad
MM   = 0.000m @ 0.036d -3.141rad
WARNING:tides.model:Tides (MTL) for 2026-03-02: low = -1.011m @ 16:56:00; high = 1.040m @ 10:23:00
```

Looking at [the buoy's MLT observations for the forecasted day](https://tidesandcurrents.noaa.gov/waterlevels.html?id=9414290&units=metric&bdate=20260302&edate=20260302&timezone=LST&datum=MTL&interval=6&action=), we find extrema of `low = -1.062m @ 16:42` and `high = 0.984m @ 10:36`, meaning the model predictions are high by `51 mm` and `56 mm` respectively and offset in time by `+14` and `-13 minutes` respectively.  This is a bit better than the NOAA harmonic model which under predicts extrema by `76 mm` and `110 mm` respectively and is offset in time by `+6` and `-18 minutes` respectively.

## Is the model any good?

The model trained on 48 months of observations (between `2021-12` and `2025-11`) from buoy `9414290` at the San Francisco Golden Gate Bridge and validated on observations from `2025-12` achieves a mean absolute error of `85 mm` and a bias of `+6 mm`.  For a midpoint within the validation month, the prediction is high by `23 mm`.  Notably, the reported uncertainty in measurements for the validation month have a mean of `52±19 mm`.  There is an `e2e` make target, but below I call the entrypoint explicitly for reproducibility.

```shell
uv run python -m tides --number-of-training-months=48 --model-name=FittedHarmonicModel --station-number=9414290 --verbose e2e --test-month=2025-12 ;
INFO:tides.model:Fetching historical (MTL) tide levels for station_number=9414290 between 2021-12 and 2025-11 (inclusive) for training
INFO:tides.model:Received 350_640 data points with 14_179 NaN's
INFO:tides.model:Fitting FittedHarmonicModel model with parameters={'learning_rate': 3}
INFO:tides.model:Found historical mean[MTL] of 0.0732m and harmonics:
M2   = 0.540m @ 1.932d -0.790rad
K1   = 0.393m @ 1.003d +2.459rad
O1   = 0.255m @ 0.930d -0.924rad
S2   = 0.135m @ 2.000d +2.506rad
N2   = 0.116m @ 1.896d +1.489rad
P1   = 0.106m @ 0.997d +2.180rad
K2   = 0.056m @ 2.005d -0.299rad
Q1   = 0.045m @ 0.893d +1.022rad
MK3  = 0.024m @ 2.935d +0.990rad
NU2  = 0.022m @ 1.901d +0.763rad
J1   = 0.022m @ 1.039d +0.288rad
M4   = 0.021m @ 3.865d -2.696rad
S1   = 0.020m @ 1.000d +2.201rad
2MK3 = 0.019m @ 2.862d -2.132rad
L2   = 0.018m @ 1.969d +0.256rad
OO1  = 0.018m @ 1.076d +2.259rad
M1   = 0.018m @ 0.966d -1.857rad
2N2  = 0.017m @ 1.860d -2.546rad
T2   = 0.014m @ 1.997d +2.217rad
MS4  = 0.012m @ 3.932d +0.607rad
LAM2 = 0.012m @ 1.964d +1.395rad
RHO  = 0.011m @ 0.898d +0.202rad
MF   = 0.011m @ 0.073d +3.141rad
MSF  = 0.011m @ 0.068d -0.000rad
MN4  = 0.010m @ 3.828d -0.426rad
2SM2 = 0.006m @ 2.068d +2.503rad
MU2  = 0.005m @ 1.865d -2.130rad
SSA  = 0.005m @ 0.005d -0.450rad
SA   = 0.004m @ 0.003d +2.010rad
M3   = 0.004m @ 2.898d -2.129rad
M6   = 0.003m @ 5.797d +2.375rad
R2   = 0.003m @ 2.003d -2.598rad
2Q1  = 0.003m @ 0.857d +2.820rad
S6   = 0.000m @ 6.000d -1.979rad
S4   = 0.000m @ 4.000d -1.608rad
M8   = 0.000m @ 7.729d -2.644rad
MM   = 0.000m @ 0.036d -3.141rad
INFO:tides.model:Fetching historical (MTL) tide levels for station_number=9414290 between 2025-12-01 and 2025-12-31 (inclusive) for validation
INFO:tides.model:Received 7_440 data points with 0 NaN's
WARNING:tides.model:Out of sample: MAE=85 mm; BIAS=+6 mm
WARNING:tides.model:At `2025-12-16 04:00:00` predicted=0.072m; observed=0.049m
```

We can put this performance in context be comparing with a model which fits a linear trend to the observed buoy data.  This baseline achieves a mean absolute error of `478 mm` and a bias of `-29 mm` and over-predicts at the midpoint of the validation month by `61 mm` -- quite a bit worse!

```shell
uv run python -m tides --number-of-training-months=48 --model-name=HistoricalTrendModel --station-number=9414290 --verbose e2e --test-month=2025-12 ;
INFO:tides.model:Fetching historical (MTL) tide levels for station_number=9414290 between 2021-11 and 2025-11 (inclusive) for training
INFO:tides.model:Received 357_840 data points with 14_179 NaN's
INFO:tides.model:Fitting HistoricalTrendModel model with parameters={}
INFO:tides.model:Found MTL trend of slope=0.051 mm/day and intercept=-0.93m
INFO:tides.model:Fetching historical (MTL) tide levels for station_number=9414290 between 2025-12-01 and 2025-12-31 (inclusive) for validation
INFO:tides.model:Received 7_440 data points with 0 NaN's
WARNING:tides.model:Out of sample: MAE=478 mm; BIAS=-29 mm
WARNING:tides.model:At `2025-12-16 04:00:00` predicted=0.110m; observed=0.049m
```

Interestingly, the trend model indicates a rising tide of `~19 mm/y`.

## What is the model approach?

The model which I found to perform best is one which fits the [37 tidal constituent harmonics](https://en.wikipedia.org/wiki/Theory_of_tides#Tidal_constituents) to observed buoy data.  While it fixes the theoretical frequency for each harmonic, the phases and amplitudes are fitted to the data using binary search and gradient descent respectively (see comments for more discussion).  Performance easily surpassed the baselines of a historical linear trend, the historical mean, and the most-recently-observed value.  I was hoping to include [NOAA's fitted harmonics](https://tidesandcurrents.noaa.gov/harcon.html?unit=0&timezone=1&id=9414290&name=San+Francisco&state=CA) as a baseline, but for the life of me I could figure out how to use the tabulated phases properly.  In theory, there is one instance (i.e., epoch) in time which designates `phase=0` for all the harmonics, even with brute-force searching I was not able to find it.  This model assumes an epoch of `1970-01-01 00:00:00 UTC`.

I mainly developed against the buoy at the San Francisco Golden Gate Bridge, but found performance to be similar for the buoy near the Tidal Basin in Washington D.C. (`8594900`) and in Pearl Harbor Hawaii (`1612401`) -- albeit with difference phases and amplitudes for the harmonics.  It's likely that some buoys are more challenging to model, due to aharmonic forcing functions like river discharges after rainstorms and onshore/offshore winds.

Between 5-10% of data was missing for the buoys I experimented with, so models had to be designed to be robust with missing data.  My approach was to simply drop `NaN`'s (e.g., with `numpy.nanmean` instead of `numpy.mean`) and although this approach is straightforward, it likely introduces error and bias due to systematic patterns in missings.  Fourier transorm libraries were not an option because I couldn't find any which handled missing data.  When implementing [N-BEATS](https://arxiv.org/abs/1905.10437), I avoided `NaN`'s by filtering out training periods with missing data -- but performance suffered from having a biased and limited dataset.

Data is ingested from [the free NOAA API](https://api.tidesandcurrents.noaa.gov/api/prod/), which provides buoy observations and predictions at six minute intervals.  The data has a notion of preliminary and verified data, where every calendar month, the previous calendar month's data is inspected.  The SLA is that a calendar month's data will be verified by the following following month (e.g., January is verified by March).  However, the ingestion logic takes no steps to differentiate between these qualities of data and uses their values as-is.

## How does the library work?

### Forecasting as a python package

1. Install the package:

```shell
uv run --with=git+https://github.com/jeffseif/tides.git python
```

2. Import and run it:

```python
import datetime

import tides.model
from numpy.typing import NDArray

levels: NDArray = tides.model.forecast_for_day(
    day=datetime.date.today(),
    model_cls=tides.model.FittedHarmonicModel,
    number_of_training_months=48,
    station_number=9414290,
)
...
```

### Forecasting from the command line

```shell
make forecast
WARNING:tides.model:Tides (MTL) for 2026-03-02: low = -1.011m @ 16:56:00; high = 1.040m @ 10:23:00
```

### Validating the model

```shell
make e2e
WARNING:tides.model:Out of sample: MAE=77 mm; BIAS=+42 mm
WARNING:tides.model:At `2026-02-14 16:00:00` predicted=-0.927m; observed=-0.960m
```
