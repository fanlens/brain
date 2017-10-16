"""
Transformer and tools to convert texts into their fingerprints. A fingerprint is a list of indexes in a 256*256 space
"""
from typing import Iterable, List, Optional, Any, Dict, cast

import numpy
import retinasdk
from retinasdk.model.fingerprint import Fingerprint as CorticalFingerprint
from scipy.sparse import lil_matrix, spmatrix
from sklearn.base import TransformerMixin

from common.config import get_config

TFingerprint = List[int]

_RETINA = retinasdk.FullClient(get_config().get("CORTICAL", "api_key"))


def _to_fingerprint(fingerprint: CorticalFingerprint) -> TFingerprint:
    positions = fingerprint.positions
    assert isinstance(positions, list)
    assert all(isinstance(idx, int) for idx in positions)
    return cast(TFingerprint, positions)


def get_fingerprints(texts: Iterable[str]) -> Iterable[TFingerprint]:
    """
    Get fingerprints for a batch of texts.
    :param texts: text batch
    :return: list of fingerprints
    """
    return map(_to_fingerprint, _RETINA.getFingerprintsForTexts(texts))


def get_fingerprint(text: str) -> TFingerprint:
    """
    Get fingerprint for text.
    :param text: the text
    :return: the fingerprint
    """
    return _to_fingerprint(_RETINA.getFingerprintForText(text))


def unpack_fingerprint(fingerprint: TFingerprint) -> numpy.ndarray:
    """
    Transform a fingerprint (list of indexes) into a dense numpy array.
    :param fingerprint: index list
    :return: dense numpy array
    """
    sample_1d = numpy.zeros(128 * 128, dtype=numpy.float64)
    sample_1d[numpy.array(fingerprint, dtype=numpy.uint16)] = 1.0
    return sample_1d


def sparsify_fingerprint(fingerprint: TFingerprint) -> spmatrix:
    """
    Transform a fingerprint (list of indexes) into a sparse numpy array.
    :param fingerprint: index list
    :return: sparse numpy array
    """
    sparse = lil_matrix((1, 128 * 128), dtype=numpy.float64)
    sparse[0, fingerprint] = 1.0
    return sparse


class DenseFingerprintTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument, no-self-use
    """Transform list of fingerprints into dense numpy arrays"""

    def transform(self, X: Iterable[TFingerprint], y: Optional[numpy.ndarray] = None,
                  **transform_params: Any) -> numpy.ndarray:
        """
        :param X: source of fingerprints
        :param y: unused
        :param transform_params: unused
        :return: a numpy array containing the densely unpacked fingerprints
        """
        return numpy.array([unpack_fingerprint(x) for x in X], dtype=numpy.float64)

    def fit(self, X: Iterable[TFingerprint], y: Optional[numpy.ndarray] = None,
            **fit_params: Any) -> 'DenseFingerprintTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[None, None]:
        """unused"""
        return {}

    def set_params(self, **_: Any) -> None:
        """unused"""
        pass


class SparseFingerprintTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument, no-self-use
    """Transform list of fingerprints into sparse numpy arrays"""

    def transform(self, X: Iterable[TFingerprint], y: Optional[numpy.ndarray] = None,
                  **transform_params: Any) -> spmatrix:
        """
        :param X: source of fingerprints
        :param y: unused
        :param transform_params: unused
        :return: a numpy array containing the sparsely unpacked fingerprints
        """
        sparse = lil_matrix((len(list(X)), 128 * 128), dtype=numpy.float64)
        for idx, x in enumerate(X):
            sparse[idx, x] = 1.0
        return sparse

    def fit(self, X: Iterable[TFingerprint], y: Optional[numpy.ndarray] = None,
            **fit_params: Any) -> 'SparseFingerprintTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[None, None]:
        """unused"""
        return {}

    def set_params(self, **_: Any) -> None:
        """unused"""
        pass
