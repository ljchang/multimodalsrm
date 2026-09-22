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


def clamp(value, low, high):
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return low
    return value
