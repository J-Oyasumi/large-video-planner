import tensorflow as tf


def _try_decode_utf8(x):
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            # Keep binary payloads (e.g. encoded images) as bytes.
            return x
    return x


def _decode_nested_strings(x):
    if isinstance(x, list):
        return [_decode_nested_strings(v) for v in x]
    return _try_decode_utf8(x)


def recursive_cast_to_numpy(obj):
    if isinstance(obj, tf.Tensor):
        if obj.dtype == tf.string:
            raw = obj.numpy()
            if obj.ndim == 0:
                return _try_decode_utf8(raw)
            return _decode_nested_strings(raw.tolist())
        else:
            # Convert non-string tensors to numpy arrays
            return obj.numpy()
    elif isinstance(obj, dict):
        # Recursively handle dictionary values
        return {key: recursive_cast_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively handle list elements
        return [recursive_cast_to_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively handle tuple elements
        return tuple(recursive_cast_to_numpy(item) for item in obj)
    else:
        # Return the object as-is if it's not a tf.Tensor
        return obj
