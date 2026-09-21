"""A public random seed must survive Python's per-process hash randomization."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_fit_seed_is_independent_of_python_hash_seed():
    root = Path(__file__).resolve().parents[1]
    program = """
import json
import warnings
import numpy as np
from multimodalsrm import MultimodalSRM
from multimodalsrm import TimeSeries
t=np.arange(10.)
data={s:{'brain':TimeSeries(np.sin(t[:,None]+i),t),
         'rating':TimeSeries(np.cos(t[:,None]+i),t)} for i,s in enumerate(['a','b'])}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m=MultimodalSRM(features=2,latent_dt=1,max_iter=3,random_state=13).fit(data)
print(json.dumps({'brain':m.loadings_['a']['brain'].tolist(),'rating':m.loadings_['a']['rating'].tolist(),'objective':m.objective_history_}))
"""
    outputs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(result.stdout))
    for key in outputs[0]:
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], rtol=1e-12, atol=1e-12)
