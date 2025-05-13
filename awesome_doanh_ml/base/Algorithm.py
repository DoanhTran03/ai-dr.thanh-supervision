class Algorithm:
    def __init__(self, name, algorithm):
        self.name = name
        self.algorithm = algorithm
        if algorithm is not None:
            self.estimator = self.algorithm()
        else:
            self.estimator = []

    def fit (self, train_data):
        self.estimator.fit(train_data)

    def predict(self, test_data):
        return self.estimator.predict(test_data)

    def fit (self, x, y):
        return self.estimator.fit(x, y)

    def predict_proba(self, data):
        return self.estimator.predict_proba(data)