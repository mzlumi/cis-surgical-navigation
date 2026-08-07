import numpy as np

import cisnav


def test_package_imports() -> None:
    assert cisnav.__version__


def test_numpy_available() -> None:
    assert np.allclose(np.eye(3) @ np.ones(3), np.ones(3))
