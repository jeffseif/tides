import csv
import datetime
import gzip
import logging
import os

import requests  # type: ignore[import-untyped]

import tides.month


logger = logging.getLogger(__name__)


CACHE_DIR = ".cache"
YYYYMMDDHHMM = "%Y-%m-%d %H:%M"
YYMMDD = "%Y%m%d"


LevelType = dict[str, float]


def get_levels_for_day_span_station(
    day_span: tuple[datetime.date, datetime.date], station_number: int
) -> list[LevelType]:
    begin_date, end_date = day_span
    if end_date >= (today := datetime.date.today()):
        raise ValueError(f"{end_date=:} >= {today=:}")
    logger.info(f"Fetching data for {begin_date:} to {end_date:} (inclusive)")
    response = requests.request(
        method="GET",
        params={
            "begin_date": begin_date.strftime(YYMMDD),
            "end_date": end_date.strftime(YYMMDD),
            "datum": "MTL",
            "format": "json",
            "product": "water_level",
            "station": station_number,
            "time_zone": "GMT",
            "units": "metric",
        },
        url="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
    )
    data = response.json()
    # https://api.tidesandcurrents.noaa.gov/api/prod/responseHelp.html
    return [
        {
            "dt": datetime.datetime.strptime(level_dict["t"], YYYYMMDDHHMM)
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp(),
            "level": float(level_dict["v"]) if level_dict["v"] else float("nan"),
            "std": float(level_dict["s"]) if level_dict["s"] else float("nan"),
        }
        for level_dict in data["data"]
    ]


FIELDNAMES = ("dt", "level", "std")


def load_for_month_station(
    month: tides.month.Month, station_number: int
) -> list[LevelType]:
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = f"{CACHE_DIR:s}/{station_number:d}:{month}.csv.gz"
    if os.path.exists(path):
        logger.debug(f"Loading cached data from {path=:s}")
        with gzip.open(filename=path, mode="rt") as fp:
            return list(
                csv.DictReader(
                    f=fp, fieldnames=FIELDNAMES, quoting=csv.QUOTE_NONNUMERIC
                )  # type: ignore
            )
    else:
        data = get_levels_for_day_span_station(
            day_span=(month.fdom, month.ldom),
            station_number=station_number,
        )
        logger.debug(f"Caching levels to {path=:s}")
        with gzip.open(filename=path, mode="wt") as fp:
            csv.DictWriter(f=fp, fieldnames=FIELDNAMES).writerows(data)
        return data


def load_for_month_span_station(
    start: tides.month.Month, end: tides.month.Month, station_number: int
) -> list[LevelType]:
    return [
        level
        for month in tides.month.iter_month(start=start, end=end)
        for level in load_for_month_station(month=month, station_number=station_number)
    ]
