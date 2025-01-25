import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class GMMNaiveBayes:
    def __init__(self, n_components=5):
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
        self.numeric_cols = None

    def _resolve_numeric_cols(self, X, numeric_cols):
        """
        Resolve numeric columns from names or indices to actual indices.

        Parameters:
        - X: array-like or DataFrame, input data.
        - numeric_cols: list of str or int, column names or indices for numeric features.

        Returns:
        - numeric_indices: list of int, resolved numeric column indices.
        """
        if isinstance(X, pd.DataFrame):
            # If X is a DataFrame, resolve column names to indices
            if all(isinstance(col, str) for col in numeric_cols):
                return [X.columns.get_loc(col) for col in numeric_cols]
            elif all(isinstance(col, int) for col in numeric_cols):
                return numeric_cols
            else:
                raise ValueError("numeric_cols must be a list of all strings or all integers.")
            
        elif isinstance(X, np.ndarray):
            # If X is a NumPy array, assume numeric_cols are indices
            if all(isinstance(col, int) for col in numeric_cols):
                return numeric_cols
            else:
                raise ValueError("If X is a NumPy array, numeric_cols must be a list of integers.")
        else:
            raise TypeError("X must be either a NumPy array or a Pandas DataFrame.")

    def fit(self, X, y, numeric_cols):
        """
        Fit the Bayesian GMM Naive Bayes classifier.

        Parameters:
        - X: array-like of shape (n_samples, n_features), training data.
        - y: array-like of shape (n_samples,), class labels.
        - numeric_cols: list of column names or indices for numeric features.
        """
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = np.unique(y_encoded)

        # Resolve numeric columns to indices
        self.numeric_cols = self._resolve_numeric_cols(X, numeric_cols)

        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array for processing

        for cls in self.classes_:
            cls_idx = np.where(y_encoded == cls)[0]
            X_cls = X[cls_idx]

            # Fit a Bayesian Gaussian Mixture Model for numeric features for each class
            bgmm = BayesianGaussianMixture(n_components=self.n_components, random_state=42)
            bgmm.fit(X_cls[:, self.numeric_cols])

            # Exclude numeric columns when calculating mean probabilities for categorical features
            categorical_cols = [i for i in range(X.shape[1]) if i not in self.numeric_cols]
            # Ensure all categorical columns are one-hot encoded
            for col in categorical_cols:
                unique_values = np.unique(X_cls[:, col])
                # print("unique_values: ", unique_values)
                # # print(np.array_equal(unique_values, [0, 1]))
                # # print(np.array_equal(unique_values, [0., 1.]))
                # print(set(unique_values))
                # print(set([0, 1]))
                # print(set(unique_values) <= set([0, 1]))
                # # if not (np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0., 1.])):
                if not (set(unique_values) <= set([0, 1]) or set(unique_values) <= set([0., 1.])):
                    raise ValueError(f"Categorical column at index {col} is not one-hot encoded.")
            # Calculate mean probabilities for one-hot encoded categorical features
            categorical_probs = np.mean(X_cls[:, categorical_cols], axis=0)

            self.models_[cls] = {
                'gmm': bgmm,
                'categorical_probs': categorical_probs  # Mean probabilities for one-hot columns
            }

            '''
            X_cls contains all the samples in the dataset belonging to a specific class cls.
            np.mean(X_cls, axis=0) calculates the mean for each column (feature) across all samples in X_cls.
            E.g: 
                categorical_data = np.array([
                [1, 0, 0],  # Category A
                [1, 0, 0],  # Category A
                [0, 1, 0],  # Category B
                [0, 0, 1],  # Category C
                [0, 0, 1],  # Category C
                [0, 1, 0],  # Category B
                ])

                y = np.array([0, 0, 0, 1, 1, 1])  # Class labels

            Class 0 categorical probabilities:
                [0.66666667 0.33333333 0.        ]
            Class 1 categorical probabilities:
                [0.         0.33333333 0.66666667]
            '''

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
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array for processing

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes_):
            model = self.models_[cls]
            gmm = model['gmm']
            categorical_probs = model['categorical_probs']
            prior = self.priors_[cls]

            # Compute likelihood for numeric features using Bayesian GMM
            numeric_likelihood = np.exp(gmm.score_samples(X[:, self.numeric_cols]))

            # Compute likelihood for one-hot encoded categorical features
            categorical_cols = [i for i in range(X.shape[1]) if i not in self.numeric_cols]
            cat_likelihood = np.prod(
                np.where(X[:, categorical_cols] == 1, categorical_probs, 1 - categorical_probs), axis=1
            )

            # Posterior probability
            probs[:, i] = numeric_likelihood * cat_likelihood * prior

        # Normalize to get probabilities
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs /= np.where(probs_sum == 0, 1e-9, probs_sum)  # Handle division by zero
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




if __name__ == "__main__":
    # Example usage:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create a synthetic dataset
    X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=42)


    # Simulate one-hot encoded categorical data by adding binary features
    categorical_cols = [f"cat_{i}" for i in range(3)]
    one_hot_data = np.random.randint(0, 2, size=(X.shape[0], len(categorical_cols)))
    X = np.hstack([X, one_hot_data])

    # Create DataFrame
    columns = [f"num_{i}" for i in range(10)] + categorical_cols
    X_df = pd.DataFrame(X, columns=columns)

    # Define numeric columns by name
    numeric_cols = [f"num_{i}" for i in range(10)]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

    print("X_train shape:", X_train.shape)
    print(X_train.head())
    print("y_train shape:", y_train.shape)
    print(y_train[:5])

    # Fit and predict using GMM Naive Bayes
    clf = GMMNaiveBayes(n_components=20)
    clf.fit(X_train, y_train, numeric_cols=numeric_cols)
    y_pred = clf.predict(X_test)
    print("Accuracy (GMM_nb):", np.mean(y_pred == y_test))

    clf = GMMNaiveBayes(n_components=20)
    clf.fit(X_train[numeric_cols], y_train, numeric_cols=numeric_cols)
    y_pred = clf.predict(X_test[numeric_cols])
    print("Accuracy (GMM_nb) - only numeric:", np.mean(y_pred == y_test))


    clf = GMMNaiveBayes(n_components=20)
    clf.fit(X_train, y_train, numeric_cols=columns)
    y_pred = clf.predict(X_test)
    print("Accuracy (GMM_nb) indiscrimate:", np.mean(y_pred == y_test))



    # Fit and predict using Naive Bayes Gaussian
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train[numeric_cols], y_train)
    y_pred = clf.predict(X_test[numeric_cols])

    print("Accuracy (GaussianNB - only numeric):", np.mean(y_pred == y_test))

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy (GaussianNB - numeric & cat):", np.mean(y_pred == y_test))


    # Fit and predict using Naive Bayes Multinomial
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()

    # Convert negative values to positive
    X_train = X_train - X_train.min()
    X_test = X_test - X_test.min()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy (MultinomialNB):", np.mean(y_pred == y_test))


    # Fit and predict using Naive Bayes categorical
    from sklearn.naive_bayes import CategoricalNB
    clf = CategoricalNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy (CategoricalNB):", np.mean(y_pred == y_test))
