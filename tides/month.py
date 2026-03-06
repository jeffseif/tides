import collections.abc
import dataclasses
import datetime
import typing


@dataclasses.dataclass(order=True)
class Month:
    year: int
    month: int

    @classmethod
    def today(cls) -> typing.Self:
        today = datetime.date.today()
        return cls(month=today.month, year=today.year)

    @classmethod
    def from_str(cls, s: str) -> typing.Self:
        yyyy, _, mm = s.partition("-")
        return cls(month=int(mm), year=int(yyyy))

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def fdom(self) -> datetime.date:
        return datetime.date(year=self.year, month=self.month, day=1)

    def __add__(self, other: int) -> Month:
        month, year = self.month + other, self.year
        while month < 1:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1
        return Month(month=month, year=year)

    def __sub__(self, other: int) -> Month:
        return self + -other

    def __next__(self) -> Month:
        return self + 1

    @property
    def ldom(self) -> datetime.date:
        return next(self).fdom - datetime.timedelta(days=1)


def iter_month(start: Month, end: Month) -> collections.abc.Iterator[Month]:
    month = start
    while month <= end:
        yield month
        month += 1
