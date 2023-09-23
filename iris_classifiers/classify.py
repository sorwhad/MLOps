from sklearn.svm import SVC


def classify_iris(data, target, classifier="SVM"):
    """Docstring."""
    if classifier == "SVM":
        clf = SVC()
        clf.fit(data, target)
        return clf.predict(data)
    else:
        raise NotImplementedError("Other classifiers are not available yet.")
