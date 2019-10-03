import numpy as np


def _is_numeric_array_like(value):
    """Returns True if the input is a valid float array.
    """
    if isinstance(value, np.ndarray):
        return True if value.dtype == float else False

    else:
        # value is not array, try to convert it to numpy float array
        try:
            value = np.asarray(value, dtype=float)
        except ValueError:
            # conversion failed, input can't be converted to float array
            return False
        finally:
            # conversion successful
            return True
