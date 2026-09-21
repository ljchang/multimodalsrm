"""Content identity for a frozen group reference, independent of object address."""

import hashlib
import json

from . import _archive


def training_reference_id(model):
    arrays = {}
    state = _archive.encode(
        dict(
            training=model.training_data_,
            parameters=model.training_fit_["map_parameters"],
            configuration=model.training_fit_["configuration"],
        ),
        arrays,
    )
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode())
    for key, array in arrays.items():
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
