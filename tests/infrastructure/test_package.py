"""These tests run against installed distributions outside the checkout too."""

import importlib.metadata
import subprocess
import sys

from packaging.requirements import Requirement


def test_installed_version_and_metadata_agree():
    import multimodalsrm

    assert multimodalsrm.__version__ == importlib.metadata.version("multimodalsrm")
    metadata = importlib.metadata.metadata("multimodalsrm")
    assert metadata["License-Expression"] == "MIT"
    assert all(Requirement(value).url is None for value in metadata.get_all("Requires-Dist", []))


def test_import_does_not_load_optional_or_legacy_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import multimodalsrm; "
            "from multimodalsrm import MultimodalSRM, TimeSeries, Response; "
            "from multimodalsrm.bayesian import BayesianMultimodalSRM; "
            "assert MultimodalSRM.__module__ == 'multimodalsrm.estimator'; "
            "assert BayesianMultimodalSRM.__module__ == 'multimodalsrm.bayesian.model'; "
            "assert not any(n.startswith('personalized_srm') for n in sys.modules); "
            "assert not {'jax', 'numpyro', 'arviz', 'nltools', 'torch'} & sys.modules.keys()",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
