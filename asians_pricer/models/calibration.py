"""
Thin QuantLib wrapper to calibrate Heston parameters to vanilla option quotes.

The calibrator keeps the QuantLib dependency optional so the rest of the
package can run in environments without QuantLib installed.
"""

from datetime import date
from typing import Iterable, Sequence, Tuple

from .heston import HestonParams

try:
    import QuantLib as ql  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ql = None


class HestonCalibrator:
    """
    Calibrate Heston parameters to vanilla option quotes using QuantLib helpers.
    """

    def __init__(self, valuation_date: date, spot_price: float, risk_free_rate: float):
        """
        Prepare QuantLib term structures and handles needed for calibration.

        Args:
            valuation_date: Date of calibration in either datetime.date or QuantLib format.
            spot_price: Observed spot/futures level for the underlying.
            risk_free_rate: Continuously compounded risk-free rate used for discounting.
        """
        if ql is None:
            raise ImportError(
                "QuantLib is required for calibration; install QuantLib-Python to use HestonCalibrator."
            )

        self.valuation_date = self._to_ql_date(valuation_date)
        self.spot = ql.QuoteHandle(ql.SimpleQuote(float(spot_price)))

        self.r_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.valuation_date, risk_free_rate, ql.Actual365Fixed())
        )
        # For futures the drift under Q is zero, so we set dividend yield = risk free.
        self.q_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.valuation_date, risk_free_rate, ql.Actual365Fixed())
        )

    @staticmethod
    def _to_ql_date(value) -> "ql.Date":
        """
        Convert a datetime.date or QuantLib Date to a QuantLib Date object.

        Returns:
            QuantLib ``Date`` instance corresponding to the input.

        Raises:
            TypeError if the input is not a supported date representation.
        """
        if isinstance(value, ql.Date):
            return value
        if isinstance(value, date):
            return ql.Date(value.day, value.month, value.year)
        raise TypeError("valuation_date must be datetime.date or QuantLib Date")

    def calibrate(
        self, market_quotes: Iterable[Tuple[float, object, float]]
    ) -> HestonParams:
        """
        Calibrate to an iterable of (strike, expiry, implied_vol) observations.

        Args:
            market_quotes: Iterable of tuples where expiry may be a QuantLib Date,
                datetime.date, or float year fraction.

        Returns:
            ``HestonParams`` calibrated by least-squares fitting of QuantLib helpers.
        """
        process = ql.HestonProcess(
            self.r_ts,
            self.q_ts,
            self.spot,
            v0=0.04,
            kappa=1.0,
            theta=0.04,
            sigma=0.5,
            rho=-0.3,
        )
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)

        calendar = ql.Australia()
        helpers = []
        for strike, expiry, vol in market_quotes:
            expiry_period = self._to_period(expiry)
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(float(vol)))
            helper = ql.HestonModelHelper(
                expiry_period,
                calendar,
                self.spot.value(),
                float(strike),
                vol_handle,
                self.r_ts,
                self.q_ts,
            )
            helper.setPricingEngine(engine)
            helpers.append(helper)

        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        model.calibrate(
            helpers,
            lm,
            ql.EndCriteria(
                maxIterations=500,
                maxStationaryStateIterations=50,
                rootEpsilon=1e-8,
                functionEpsilon=1e-8,
                gradientNormEpsilon=1e-8,
            ),
        )
        theta_c, kappa_c, sigma_c, rho_c, v0_c = model.params()
        return HestonParams(v0=v0_c, kappa=kappa_c, theta=theta_c, sigma=sigma_c, rho=rho_c)

    def _to_period(self, expiry) -> "ql.Period":
        """
        Convert various expiry representations to a QuantLib Period.

        Accepts QuantLib Period/Date, datetime.date, or numeric year fractions and
        maps them into day-based periods for calibration helpers.

        Returns:
            QuantLib ``Period`` representing the supplied expiry.
        """
        if isinstance(expiry, ql.Period):
            return expiry
        if isinstance(expiry, (ql.Date, date)):
            exp_date = self._to_ql_date(expiry)
            return ql.Period(exp_date - self.valuation_date, ql.Days)
        # fall back to treating numeric as year fraction
        if isinstance(expiry, (int, float)):
            days = int(round(float(expiry) * 365.0))
            return ql.Period(days, ql.Days)
        raise TypeError("expiry must be QuantLib Date/Period, datetime.date, or a year fraction")
