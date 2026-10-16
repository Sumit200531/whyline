"""whyline — explain why a metric moved.

>>> import whyline as wl
>>> data = wl.synth.demo().frame
>>> result = wl.explain(data, wl.Ratio("conversions", "sessions", name="conversion rate", percent=True),
...                     dimensions=["country", "device", "channel", "user_type"], time="week")
>>> print(result)          # doctest: +SKIP
"""

from . import synth
from .explain import Cause, Explanation, explain
from .metrics import Ratio, Sum

__all__ = ["explain", "Explanation", "Cause", "Ratio", "Sum", "synth"]
__version__ = "0.1.0"
