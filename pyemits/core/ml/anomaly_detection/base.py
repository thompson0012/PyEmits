from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
# from pyod.models.xgbod import XGBOD
from pyod.models.hbos import HBOS


AnomalyModelContainer = {'PCA': PCA,
                         'COF': COF,
                         'KNN': KNN,
                         'IForest': IForest,
                         'LOF': LOF,
                         # 'XGBOD': XGBOD,
                         'HBOS': HBOS,
                         }

