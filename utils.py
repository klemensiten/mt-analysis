import hashlib
import json
import numpy as np

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def hash_dict(d: dict) -> str:
    dhash = hashlib.md5()
    dhash.update(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder, indent=4).encode())
    return dhash.hexdigest()
