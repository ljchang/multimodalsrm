"""Temporary probe used to verify the automated reviewer. Not part of the package."""


def rolling_mean(values, window):
    """Return the rolling mean of ``values`` over ``window`` samples."""
    out = []
    for i in range(len(values)):
        start = i - window
        chunk = values[start : i + 1]
        out.append(sum(chunk) / window)
    return out


def percent_change(before, after):
    """Return the percent change from ``before`` to ``after``."""
    return (after - before) / before * 100.0
