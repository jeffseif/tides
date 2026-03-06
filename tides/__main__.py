import argparse
import datetime
import logging

import tides.month
import tides.ingest
import tides.model

logger = logging.getLogger(__name__)


DEFAULT_STATION_NUMBER = 9414290


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        choices=tides.model.NAME_TO_MODEL,
        default=tides.model.FittedHarmonicModel.__name__,
    )
    parser.add_argument("--number-of-training-months", type=int, default=48)
    parser.add_argument("--station-number", type=int, default=DEFAULT_STATION_NUMBER)
    parser.add_argument("--verbose", "-v", action="count", default=0)

    subparser = parser.add_subparsers(required=True)
    e2e = subparser.add_parser("e2e")
    e2e.set_defaults(func=tides.model.train_and_validate_for_test_month)
    e2e.add_argument(
        "--test-month",
        type=tides.month.Month.from_str,
        default=tides.month.Month.today() - 1,
    )
    forecast = subparser.add_parser("forecast")
    forecast.set_defaults(func=tides.model.forecast_for_day)
    forecast.add_argument(
        "--day",
        type=lambda s: datetime.datetime.fromisoformat(s).date(),
        default=datetime.date.today(),
    )
    args = parser.parse_args()
    kwargs = args.__dict__

    if (verbosity := kwargs.pop("verbose")) == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)

    kwargs["model_cls"] = tides.model.NAME_TO_MODEL[kwargs.pop("model_name")]
    kwargs.pop("func")(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
