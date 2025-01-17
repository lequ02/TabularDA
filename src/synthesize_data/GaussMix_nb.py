import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder

class GMMNaiveBayes:
    def __init__(self, n_components=20):
        """
        Initialize the GMM Naive Bayes classifier.

        Parameters:
        - n_components: int, maximum number of components for each Bayesian Gaussian Mixture Model.
        Depending on the data and the value of the weight_concentration_prior the model can decide to not use all the components by setting some component weights_ to values very close to zero. 
        The number of effective components is therefore smaller than n_components.
        """
        self.n_components = n_components
        self.classes_ = None
        self.models_ = {}
        self.priors_ = {}
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):
        """
        Fit the Bayesian GMM Naive Bayes classifier.

        Parameters:
        - X: array-like of shape (n_samples, n_features), training data.
        - y: array-like of shape (n_samples,), class labels.
        """
        self.classes_ = np.unique(y)
        y_encoded = self.label_encoder_.fit_transform(y)

        for cls in self.classes_:
            cls_idx = np.where(y_encoded == cls)[0]
            X_cls = X[cls_idx]

            # Fit a Bayesian Gaussian Mixture Model for each class
            bgmm = BayesianGaussianMixture(n_components=self.n_components, random_state=42)
            bgmm.fit(X_cls)
            self.models_[cls] = bgmm

            # Estimate prior probability
            self.priors_[cls] = len(cls_idx) / len(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        - X: array-like of shape (n_samples, n_features), input data.

        Returns:
        - probabilities: array-like of shape (n_samples, n_classes), predicted probabilities.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes_):
            bgmm = self.models_[cls]
            prior = self.priors_[cls]

            # Compute likelihood using Bayesian GMM
            likelihood = np.exp(bgmm.score_samples(X))

            # Posterior probability
            probs[:, i] = likelihood * prior

        # Normalize to get probabilities
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        - X: array-like of shape (n_samples, n_features), input data.

        Returns:
        - labels: array-like of shape (n_samples,), predicted class labels.
        """
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(predictions)

# # Example usage:
# if __name__ == "__main__":
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split

#     # Create a synthetic dataset
#     X, y = make_classification(n_samples=500, n_features=2, n_classes=3, n_clusters_per_class=1, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Fit and evaluate Bayesian GMM Naive Bayes
#     clf = GMMNaiveBayes(n_components=5)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     print("Accuracy:", np.mean(y_pred == y_test))
