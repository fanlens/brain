"""module for a lemma based tokenizer"""

import datetime
from typing import Iterable, Optional, Any, Union, Dict

import dateutil.parser
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.base import TransformerMixin

TTimestamp = Union[int, str, datetime.datetime]


class TimeOfDayTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """
    Convert timestamps (epoch, parseable str, datetime) into an integer indicating the time of day based on a resolution
    """

    def __init__(self, dense: bool = False, resolution: int = 1) -> None:
        """
        :param dense: should a dense numpy array be returned?
        :param resolution: how many ticks per hour? default 1
        """
        self._dense = dense
        self._resolution = resolution

    def transform(self, X: Iterable[TTimestamp], y: Optional[np.ndarray] = None, **transform_params: Any) -> np.ndarray:
        """
        :param X: source of compatible timestamps
        :param y: unused
        :param transform_params: unused
        :return: numpy array of time of day ticks based on resolution
        :raises ValueError:
        Raised for invalid or unknown string format, if the provided
        :class:`tzinfo` is not in a valid format, or if an invalid date
        would be created.
        :raises OverflowError:
        Raised if the parsed date exceeds the largest valid C integer on
        your system.
        """
        num_ticks = 24 * self._resolution
        if self._dense:
            mat = np.zeros((len(list(X)), num_ticks), dtype=np.float64)
        else:
            mat = lil_matrix((len(list(X)), num_ticks), dtype=np.float64)
        for idx, x in enumerate(X):
            mat[idx, self(x)] = 1
        return mat

    def fit(self, X: Iterable[TTimestamp], y: Optional[np.ndarray] = None,
            **fit_params: Any) -> 'TimeOfDayTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Union[bool, int]]:
        """Get parameters for this Transformer."""
        return dict(dense=self._dense, resolution=self._resolution)

    def set_params(self, **params: Any) -> None:
        """Set parameters for this Transformer."""
        self._dense = params.get('dense', self._dense)
        self._resolution = params.get('resolution', self._resolution)

    def __call__(self, timestamp: TTimestamp) -> int:
        """
        :param timestamp: a epoch integer, a parseable string, a datetime.datetime
        :return: a tick for the day based on resultion, e.g. with resolution 3 -> 17:00 -> 51
        :raises ValueError:
        Raised for invalid or unknown string format, if the provided
        :class:`tzinfo` is not in a valid format, or if an invalid date
        would be created.
        :raises OverflowError:
        Raised if the parsed date exceeds the largest valid C integer on
        your system.
        """
        if isinstance(timestamp, int):
            timestamp = datetime.datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):  # assuming parsable
            timestamp = dateutil.parser.parse(timestamp)

        assert isinstance(timestamp, datetime.datetime)
        midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        minutes = (timestamp - midnight).seconds / 60
        tick = int(round(minutes / (60 / self._resolution))) % (24 * self._resolution)  # 24 -> 0 via modulo
        return tick
