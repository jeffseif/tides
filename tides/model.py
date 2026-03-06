import dataclasses
import datetime
import logging
import typing

import numpy
from numpy.typing import NDArray

import tides.harmonic
import tides.ingest
import tides.month

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Model:
    parameters: dict[str, int] = dataclasses.field(default_factory=dict)
    fitted: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, attr: str) -> typing.Any:
        if attr in self.fitted:
            return self.fitted[attr]
        elif attr in self.parameters:
            return self.parameters[attr]
        else:
            return self.__getattribute__(attr)

    def __str__(self) -> str:
        raise NotImplementedError()

    def fit(self, X: NDArray, y: NDArray) -> typing.Self:
        raise NotImplementedError()

    def predict(self, X: NDArray) -> NDArray:
        raise NotImplementedError()


class MostRecentValueModel(Model):
    def fit(self, X: NDArray, y: NDArray) -> typing.Self:
        self.fitted |= {"X": X, "y": y}
        return self

    def __str__(self) -> str:
        minimum, maximum = map(
            datetime.datetime.fromtimestamp, (min(self.X), max(self.X))
        )
        return f"Memorized levels between {minimum:} and {maximum:} local time"

    def predict(self, X: NDArray) -> NDArray:
        idxs = numpy.searchsorted(self.X, X) - 1
        return self.y[idxs]


class HistoricalMeanModel(Model):
    def fit(self, X: NDArray, y: NDArray) -> typing.Self:
        # When NaN's are present in y, they are effectively ignored
        self.fitted |= {"ymean": float(numpy.nanmean(a=y))}
        return self

    def __str__(self) -> str:
        return f"Found historical mean[MTL] of {self.ymean:.4f}m"

    def predict(self, X: NDArray) -> NDArray:
        return numpy.ones_like(X) * self.ymean


class HistoricalTrendModel(Model):
    def fit(self, X: NDArray, y: NDArray) -> typing.Self:
        # When NaN's are present in y, they are effectively ignored
        (var, cov), _ = numpy.ma.cov(
            x=numpy.ma.masked_invalid(a=X),
            y=numpy.ma.masked_invalid(a=y),
        )
        slope = cov / var
        self.fitted |= {
            "slope": float(slope),
            # Again, ignore NaN's
            "intercept": float(numpy.nanmean(a=y) - slope * numpy.nanmean(a=X)),
        }
        return self

    def __str__(self) -> str:
        slope = self.slope * 86_400 * 1e3
        intercept = self.intercept
        return f"Found MTL trend of {slope=:.3f} mm/day and {intercept=:.2f}m"

    def predict(self, X: NDArray) -> NDArray:
        return self.intercept + X * self.slope


class FittedHarmonicModel(Model):
    def fit(self, X: NDArray, y: NDArray) -> typing.Self:
        assert "learning_rate" in self.parameters
        harmonics = (
            # I tried fitting the frequency from data but found the gradient field to
            # be extremely noisy and unstable
            harmonic
            # Fit the phase first (with unity amplitude)
            .with_fitted_phase(X=X, y=y)
            # Fit the amplitude second (with the fited phase)
            .with_fitted_amplitude(learning_rate=self.learning_rate, X=X, y=y)
            for harmonic in tides.harmonic.UNFITTED_HARMONICS
        )
        self.fitted |= {
            # Order from largest to smallest amplitude
            "harmonics": sorted(harmonics, reverse=True),
            "ymean": float(numpy.nanmean(a=y)),
        }
        return self

    def __str__(self) -> str:
        lines = (
            f"Found historical mean MTL of {self.ymean=:.3f} m and harmonics:",
            *map(str, self.harmonics),
        )
        return "\n".join(lines)

    def predict(self, X: NDArray) -> NDArray:
        assert set(self.fitted).issuperset({"harmonics", "ymean"})
        components = numpy.array([harmonic.predict(X=X) for harmonic in self.harmonics])
        return components.sum(axis=0) + self.ymean


CLS_TO_PARAMETERS: dict[type[Model], dict[str, typing.Any]] = {
    MostRecentValueModel: {},
    HistoricalMeanModel: {},
    HistoricalTrendModel: {},
    FittedHarmonicModel: {"learning_rate": 3},
}
NAME_TO_MODEL = {cls.__name__: cls for cls in CLS_TO_PARAMETERS}


def get_x_y_from_levels(
    levels: list[tides.ingest.LevelType],
) -> tuple[NDArray, NDArray]:
    dt, levels = zip(*((d["dt"], d["level"]) for d in levels))  # type: ignore
    return numpy.array(dt), numpy.array(levels)


def train(
    end_month: tides.month.Month,
    model_cls: type[Model],
    number_of_months: int,
    parameters: dict[str, typing.Any],
    station_number: int,
) -> Model:
    start_month = end_month - number_of_months + 1
    logger.info(
        f"Fetching historical (MTL) tide levels for {station_number=:d} "
        f"between {start_month} and {end_month} (inclusive) for training"
    )
    X, y = get_x_y_from_levels(
        levels=tides.ingest.load_for_month_span_station(
            end=end_month,
            start=start_month,
            station_number=station_number,
        )
    )
    logger.info(f"Received {len(X):_} data points with {numpy.isnan(y).sum():_} NaN's")
    logger.info(f"Fitting {model_cls.__name__:s} model with {parameters=:}")
    model = model_cls(parameters=parameters).fit(X=X, y=y)
    logger.info(str(model))
    return model


def validate(
    model: Model,
    month: tides.month.Month,
    station_number: int,
) -> None:
    logger.info(
        f"Fetching historical (MTL) tide levels for {station_number=:d} "
        f"between {month.fdom} and {month.ldom} (inclusive) for validation"
    )
    X, y = get_x_y_from_levels(
        levels=tides.ingest.load_for_month_span_station(
            end=month, start=month, station_number=station_number
        )
    )
    logger.info(f"Received {len(X):_} data points with {numpy.isnan(y).sum():_} NaN's")
    yhat = model.predict(X=X)
    mae = float(numpy.nanmean(numpy.abs(y - yhat)))
    bias = float(numpy.nanmean(y - yhat))
    logger.warning(
        f"Out of sample: MAE={mae * 1000:.0f} mm; BIAS={bias * 1000:+.0f} mm"
    )

    idx = len(X) // 2
    X0 = X[idx : idx + 1]
    dt = datetime.datetime.fromtimestamp(float(X0[0]))
    (y0,) = model.predict(X0)
    logger.warning(f"At `{dt:}` predicted={y0:.3f}m; observed={y[idx]:.3f}m")


def train_and_validate_for_test_month(
    model_cls: type[Model],
    number_of_training_months: int,
    station_number: int,
    test_month: tides.month.Month,
) -> None:
    validate(
        model=train(
            end_month=test_month - 1,
            model_cls=model_cls,
            number_of_months=number_of_training_months,
            parameters=CLS_TO_PARAMETERS[model_cls],
            station_number=station_number,
        ),
        month=test_month,
        station_number=station_number,
    )


def forecast_for_day(
    day: datetime.date,
    model_cls: type[Model],
    number_of_training_months: int,
    station_number: int,
) -> NDArray:
    if number_of_training_months < 12:
        logger.warning(
            "Providing less than a year of training data will likely result in poor model quality"
        )
    X = numpy.arange(
        (
            start_dt := datetime.datetime(
                year=day.year,
                month=day.month,
                day=day.day,
            )
        ).timestamp(),
        (start_dt + datetime.timedelta(days=1)).timestamp(),
        60,
    )
    logger.info(f"Forecasting (MTL) tide levels for {station_number=:d} for {day:}")
    yhat = train(
        end_month=tides.month.Month(year=day.year, month=day.month) - 1,
        model_cls=model_cls,
        number_of_months=number_of_training_months,
        parameters=CLS_TO_PARAMETERS[model_cls],
        station_number=station_number,
    ).predict(X=X)
    low_time, high_time = (
        datetime.datetime.fromtimestamp(timestamp=X[argminmax(yhat)]).time()
        for argminmax in (numpy.argmin, numpy.argmax)
    )
    logger.warning(
        f"Tides (MTL) for {day}: low = {min(yhat):.3f}m @ {low_time}; "
        f"high = {max(yhat):.3f}m @ {high_time}"
    )
    return yhat
