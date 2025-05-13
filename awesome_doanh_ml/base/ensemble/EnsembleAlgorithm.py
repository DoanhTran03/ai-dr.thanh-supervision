from awesome_doanh_ml.base.Algorithm import Algorithm

class EnsembleAlgorithm(Algorithm):

    def __init__(self, name, algorithm, n_estimators):
        super().__init__(name, algorithm)
        self.n_estimators = n_estimators
        self.estimators = []