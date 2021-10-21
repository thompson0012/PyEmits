"""
Why need dimensional reduction

The following is the use of dimensionality reduction in the data set:
• As data dimensions continue to decrease, the space required for data storage will also decrease.
• Low-dimensional data helps reduce calculation/training time.
• Some algorithms tend to perform poorly on high-dimensional data, and dimensionality reduction can improve algorithm availability.
• Dimensionality reduction can solve the problem of multicollinearity by removing redundant features. For example, we have two variables: "On the treadmill for a period of time
  Time spent” and “calorie consumption”. These two variables are highly correlated. The longer the time spent on the treadmill, the more calories burned.
  Naturally, the more. Therefore, it does not make much sense to store these two data at the same time, just one is enough.
• Dimensionality reduction helps data visualization. As mentioned earlier, if the dimensionality of the data is very high, the visualization will become quite difficult, while drawing two-dimensional three-dimensional
The graph of dimensional data is very simple.

Common dimensional reduction techniques:
1. missing value ratio
2. low variance filter
3. high correlation filter
4. random forest
5. backward feature elimination
6. forward feature selection
7. factor analysis
8. principle components analysis
9. independent component analysis
10. IOSMAP
11. t-SNE
12. UMAP

"""
random_state = 0
from enum import Enum


class FeatureSelection(Enum):
    @classmethod
    def missing_value_ratio(cls, threshold):
        return

    @classmethod
    def low_variance_filter(cls, threshold):
        return

    @classmethod
    def high_correlation_filter(cls, threshold):
        return

    @classmethod
    def random_forest(cls):
        from sklearn.ensemble import RandomForestRegressor
        RF = RandomForestRegressor()
        RF.fit()
        RF.feature_importances_
        return

    @classmethod
    def backward_feature_extraction(cls):
        from sklearn.linear_model import LinearRegression
        from sklearn.feature_selection import RFE
        clf = LinearRegression()
        rfe = RFE(clf, 10)
        rfe = rfe.fit_transform()
        return

    @classmethod
    def forward_feature_extraction(cls):
        from sklearn.feature_selection import f_regression
        ffs = f_regression()

        return


class ProjectionBased(Enum):
    @classmethod
    def isomap(cls):
        from sklearn.manifold import Isomap
        ISOMAP = Isomap(neighbors_algorithm=5, n_components=3, n_jobs=-1)
        ISOMAP.fit_transform()
        return

    @classmethod
    def tsne(cls):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=3, n_iter=300)
        tsne.fit_transform()
        return

    @classmethod
    def umap(cls):
        # install umap
        return


class ComponentsFactorsBased(Enum):
    @classmethod
    def factor_analysis(cls):
        from sklearn.decomposition import FactorAnalysis
        FA = FactorAnalysis(n_components=3)
        FA.fit_transform()
        return

    @classmethod
    def pca(cls):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit_transform()
        return

    @classmethod
    def ica(cls):
        from sklearn.decomposition import FastICA
        ICA = FastICA(n_components=3)
        ICA.fit_transform()
        return

    @classmethod
    def lda(cls, solver='svd', n_components=3):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        LDA = LinearDiscriminantAnalysis(solver=solver, n_components=n_components)
        LDA.fit_transform()
        return

