import dataclasses
import functools
import itertools
import logging
import typing

import numpy
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, order=True)
class Harmonic:
    amplitude_meters: float | None
    description: str
    frequency_days: float
    name: str
    phase_radians: float | None

    @classmethod
    def from_name_speed_description(
        cls, name: str, speed: float, description: str
    ) -> typing.Self:
        return cls(
            # We'll fit this in a minute
            amplitude_meters=None,
            description=description,
            # Degrees per hour -> cycles per day
            frequency_days=speed * 24 / 360,
            name=name,
            # We'll fit this in a minute
            phase_radians=None,
        )

    def __str__(self) -> str:
        return (
            f"{self.name:4s} = {self.amplitude_meters:.3f}m @ "
            f"{self.frequency_days:.3f}d {self.phase_radians:+.3f}rad"
        )

    @property
    def frequency_radians_per_second(self) -> float:
        # Cycles per day -> radians per second
        return self.frequency_days * 2 * numpy.pi / 86_400

    def with_fitted_phase(self, X: NDArray, y: NDArray) -> typing.Self:
        # During experimentation, I found gradient descent to be unstable for several
        # of the ~zero-amplitude harmonics (presumably because of the signal being
        # masked by noise).  In contrast, binary search was stable for all harmonics,
        # it ran much faster, relied on one fewer hyperparameter (just the maximum
        # change in phase for an update but not a learning rate), and it used the same
        # amount of code!
        logger.debug(f"Fitting phase for {self.name:s}")

        @functools.cache
        def get_loss(phase_radians: float) -> float:
            # At the optimal phase, the convolution of the harmonic and data is
            # maximized ... so we can use the negative of the convolution as a loss to
            # minimize.
            #
            # When NaN's are present in y, they are effectively ignored.
            #
            # Because the optimal phase is independent of amplitude, we can fit the former
            # before the latter.
            return -numpy.nanmean(
                y * numpy.cos(self.frequency_radians_per_second * X + phase_radians)
            )

        left, right = -numpy.pi, +numpy.pi
        while (right - left) > 1e-4:
            phase_radians = (left + right) / 2
            if get_loss(phase_radians=left) >= get_loss(phase_radians=right):
                left = phase_radians
            else:
                right = phase_radians
        return dataclasses.replace(self, phase_radians=phase_radians)

    def with_fitted_amplitude(
        self, learning_rate: float, X: NDArray, y: NDArray
    ) -> typing.Self:
        # Gradient descent was found to be fast and stable for all harmonics!
        logger.debug(f"Fitting amplitude for {self.name:s}")
        assert self.phase_radians is not None

        def get_grad(amplitude_meters: float) -> float:
            cosine = numpy.cos(
                self.frequency_radians_per_second * X + self.phase_radians
            )
            # Note that we handle NaN's gracefully by dropping their contribution
            # to the gradient.
            return numpy.nanmean(
                # loss = 0.5 (yhat - y) ** 2
                # yhat = a * cos
                # dloss/dyhat = yhat - y = a * cos - y
                (amplitude_meters * cosine - y)
                # dyhat/da = cos
                * cosine
            )

        # Zero is a great starting point for amplitude ... but 1e-3 may not always
        # be the best value for grad tolerance.
        amplitude_meters = 0.0
        while abs(grad := get_grad(amplitude_meters=amplitude_meters)) > 1e-3:
            amplitude_meters -= grad * learning_rate
        return dataclasses.replace(self, amplitude_meters=float(amplitude_meters))

    def predict(self, X: NDArray) -> NDArray:
        return self.amplitude_meters * numpy.cos(
            self.frequency_radians_per_second * X + self.phase_radians
        )


_NAME_SPEED_DESCRIPTIONS = (
    ("M2", 28.984104, "Principal lunar semidiurnal constituent"),
    ("S2", 30.0, "Principal solar semidiurnal constituent"),
    ("N2", 28.43973, "Larger lunar elliptic semidiurnal constituent"),
    ("K1", 15.041069, "Lunar diurnal constituent"),
    ("M4", 57.96821, "Shallow water overtides of principal lunar constituent"),
    ("O1", 13.943035, "Lunar diurnal constituent"),
    ("M6", 86.95232, "Shallow water overtides of principal lunar constituent"),
    ("MK3", 44.025173, "Shallow water terdiurnal"),
    ("S4", 60.0, "Shallow water overtides of principal solar constituent"),
    ("MN4", 57.423832, "Shallow water quarter diurnal constituent"),
    ("NU2", 28.512583, "Larger lunar evectional constituent"),
    ("S6", 90.0, "Shallow water overtides of principal solar constituent"),
    ("MU2", 27.968208, "Variational constituent"),
    ("2N2", 27.895355, "Lunar elliptical semidiurnal second-order constituent"),
    ("OO1", 16.139101, "Lunar diurnal"),
    ("LAM2", 29.455626, "Smaller lunar evectional constituent"),
    ("S1", 15.0, "Solar diurnal constituent"),
    ("M1", 14.496694, "Smaller lunar elliptic diurnal constituent"),
    ("J1", 15.5854435, "Smaller lunar elliptic diurnal constituent"),
    ("MM", 0.5443747, "Lunar monthly constituent"),
    ("SSA", 0.0821373, "Solar semiannual constituent"),
    ("SA", 0.0410686, "Solar annual constituent"),
    ("MSF", 1.0158958, "Lunisolar synodic fortnightly constituent"),
    ("MF", 1.0980331, "Lunisolar fortnightly constituent"),
    ("RHO", 13.471515, "Larger lunar evectional diurnal constituent"),
    ("Q1", 13.398661, "Larger lunar elliptic diurnal constituent"),
    ("T2", 29.958933, "Larger solar elliptic constituent"),
    ("R2", 30.041067, "Smaller solar elliptic constituent"),
    ("2Q1", 12.854286, "Larger elliptic diurnal"),
    ("P1", 14.958931, "Solar diurnal constituent"),
    ("2SM2", 31.015896, "Shallow water semidiurnal constituent"),
    ("M3", 43.47616, "Lunar terdiurnal constituent"),
    ("L2", 29.528479, "Smaller lunar elliptic semidiurnal constituent"),
    ("2MK3", 42.92714, "Shallow water terdiurnal constituent"),
    ("K2", 30.082138, "Lunisolar semidiurnal constituent"),
    ("M8", 115.93642, "Shallow water eighth diurnal constituent"),
    ("MS4", 58.984104, "Shallow water quarter diurnal constituent"),
)
UNFITTED_HARMONICS = list(
    itertools.starmap(Harmonic.from_name_speed_description, _NAME_SPEED_DESCRIPTIONS)
)
