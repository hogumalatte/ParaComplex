
import math

def round_filters(filters, width_coefficient, divisor=8):
    """Adjusts the number of filters in EfficientNet"""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Adjusts the number of repeats in EfficientNet"""
    return int(math.ceil(depth_coefficient * repeats))

def exists(val):
    """Utility function to check if a value is not None"""
    return val is not None

