"""
documentation: https://pycombo.readthedocs.io/en/latest/

"""

from combo.models.score_comb import average, maximization, median, majority_vote
from sklearn.ensemble import StackingRegressor, VotingRegressor
from typing import Literal


class EnsembleBase:
    def __init__(self,
                 ensemble_method: Literal[
                     'average', 'maximization', 'median', 'majority_vote', 'stack', 'blend', 'voting']):
        self.ensemble_method = ensemble_method


class BlendingRegressor:
    """
    References
    ----------
    https://curtis0982.blogspot.com/2019/10/stacking-and-blending-in-ml.html

    """
    pass
