"""Quickstart: explain a conversion-rate drop, then a revenue change, on the same data.

Run:  python examples/quickstart.py
"""

import whyline as wl

# Ten weeks of e-commerce funnel data with three incidents planted in the last week.
# Replace this with your own long-format DataFrame: one row per period × segment.
data = wl.synth.demo().frame
dims = ["country", "device", "channel", "user_type"]

# 1) A ratio metric: whyline separates "who showed up" (mix) from "how they behaved" (rate).
conversion = wl.Ratio("conversions", "sessions", name="conversion rate", percent=True)
result = wl.explain(data, conversion, dims, time="week")

print(result.headline(), "\n")
print(result)
result.to_html("conversion_report.html")

# 2) An additive metric on the same data.
revenue = wl.explain(data, wl.Sum("revenue"), dims, time="week")
print("\n" + revenue.headline())

# 3) Everything is also available as data.
print(result.to_frame()[["rank", "segment", "mechanism", "impact", "share", "z"]])
