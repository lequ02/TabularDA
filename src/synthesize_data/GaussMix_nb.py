import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class GMMNaiveBayes:
    def __init__(self, n_components=5, is_classification = True):
        """
        Initialize the GMM Naive Bayes classifier.

        Parameters:
        - n_components: int, maximum number of components for each Bayesian Gaussian Mixture Model.
        Depending on the data and the value of the weight_concentration_prior the model can decide to not use all the components by setting some component weights_ to values very close to zero. 
        The number of effective components is therefore smaller than n_components.
        
        - is_classification: bool, determines if the task is classification or regression.
        """
        self.n_components = n_components
        self.classes_ = None
        self.models_ = {}
        self.priors_ = {}
        self.label_encoder_ = LabelEncoder() if is_classification else None
        self.numeric_cols = None
        self.categorical_cols = None
        self.is_classification = is_classification
        self.max_iter_gmm = 500

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

        # Resolve numeric columns to indices
        self.numeric_cols = self._resolve_numeric_cols(X, numeric_cols)

        print('\n\n X in GaussMix_nb')
        print(X.head())

        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array for processing
        if isinstance(y, pd.Series):
            y = y.values  # Convert Series to NumPy array

        # Resolve categorical columns to indices
        self.categorical_cols = [i for i in range(X.shape[1]) if i not in self.numeric_cols]
        # Ensure all categorical columns are one-hot encoded
        for col in self.categorical_cols:
            unique_values = np.unique(X[:, col])
            if not (set(unique_values) <= set([0, 1]) or set(unique_values) <= set([0., 1.])):
                raise ValueError(f"Categorical column at index {col} is not one-hot encoded.")

        if self.is_classification:
            y_encoded = self.label_encoder_.fit_transform(y)
            self.classes_ = np.unique(y_encoded)

            for cls in self.classes_:
                cls_idx = np.where(y_encoded == cls)[0]
                X_cls = X[cls_idx]

                print("cls", cls)
                print("cls_idx: ", cls_idx)
                print("X_cls: ", X_cls)

                if self.numeric_cols != []:
                    # Fit a Bayesian Gaussian Mixture Model for numeric features for each class
                    bgmm = BayesianGaussianMixture(n_components=self.n_components, random_state=42, max_iter=self.max_iter_gmm)
                    bgmm.fit(X_cls[:, self.numeric_cols])
                else:
                    bgmm = None

                # Calculate mean probabilities for one-hot encoded categorical features
                categorical_probs = np.mean(X_cls[:, self.categorical_cols], axis=0)

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

        else: # Regression

            # # Fit GMM for numeric features
            # self.numeric_gmm_ = BayesianGaussianMixture(n_components=self.n_components, random_state=42, max_iter=self.max_iter_gmm)
            # self.numeric_gmm_.fit(X[:, self.numeric_cols])

            # # Calculate mean probabilities for one-hot encoded categorical features
            # self.categorical_probs_ = np.mean(X[:, self.categorical_cols], axis=0)

            # # Fit GMM for the target variable
            # y = y.reshape(-1, 1)  # Ensure y is 2D
            # self.target_gmm_ = BayesianGaussianMixture(n_components=self.n_components, random_state=42, max_iter=self.max_iter_gmm)
            # self.target_gmm_.fit(y)

            
            # Fit GMM for the target variable
            y = y.reshape(-1, 1)  # Ensure y is 2D
            self.target_gmm_ = BayesianGaussianMixture(n_components=self.n_components, random_state=42, max_iter=self.max_iter_gmm)
            self.target_gmm_.fit(y)

            print('n_components: ', self.n_components)
            print("number of components in gmm: ", self.target_gmm_.n_components)
            print("means", self.target_gmm_.means_)
            print("weights", self.target_gmm_.weights_)

            # For each component of the target GMM, fit a GMM to the numeric features and calculate mean probabilities for one-hot encoded categorical features of the corresponding samples
            for component_num in range(self.target_gmm_.n_components):
                # Get the indices of the samples that belong to the current component
                component_mask = self.target_gmm_.predict(y) == component_num
                component_idx = np.where(component_mask)[0]
                
                # Skip components with no samples
                if len(component_idx) == 0:
                    print(f"Skipping component {component_num} as it has no samples.")
                    continue
                
                X_component = X[component_idx]

                print("unique components: ", np.unique(self.target_gmm_.predict(y)))
                print(f"Checking component {component_num}: Predict mask sum: {np.sum(component_mask)}")
                print("component_idx: ", component_idx)
                print("X_component shape: ", X_component.shape)

                bgmm = None
                if self.numeric_cols:
                    # Check if there are enough samples to fit the BGMM (at least 2)
                    if len(component_idx) >= 2:
                        # Dynamically adjust n_components based on available samples
                        n_components_bgmm = min(self.n_components, len(component_idx)) # Ensure n_components does not exceed number of samples (n_components must be <= n_samples)
                        bgmm = BayesianGaussianMixture(n_components=n_components_bgmm, random_state=42, max_iter=self.max_iter_gmm)
                        bgmm.fit(X_component[:, self.numeric_cols])
                    else:
                        print(f"Not enough samples ({len(component_idx)}) in component {component_num} to fit BGMM.")
                
                # Calculate mean probabilities for one-hot encoded categorical features
                categorical_probs = np.mean(X_component[:, self.categorical_cols], axis=0)

                self.models_[component_num] = {
                    'gmm': bgmm,
                    'categorical_probs': categorical_probs
                }


    def predict_proba(self, X):
        """
        Predict class probabilities for the input data (classification).

        Parameters:
        - X: array-like of shape (n_samples, n_features), input data.

        Returns:
        - probabilities: array-like of shape (n_samples, n_classes), predicted probabilities.
        """
        if not self.is_classification:
            raise ValueError("predict_proba() is only available for classification tasks.")
        
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
            # numeric_likelihood = np.exp(gmm.score_samples(X[:, self.numeric_cols]))

            if self.numeric_cols != []:
                log_probs = gmm.score_samples(X[:, self.numeric_cols])
                numeric_likelihood = np.exp(log_probs - np.max(log_probs))  # Stabilized likelihood
                # use log_probs - np.max(log_probs) instead of log_probs to avoid overflow in exp()
            else:
                numeric_likelihood = np.ones(n_samples)


            # Compute likelihood for one-hot encoded categorical features
            # categorical_cols = [i for i in range(X.shape[1]) if i not in self.numeric_cols]
            cat_likelihood = np.prod(
                np.where(X[:, self.categorical_cols] == 1, categorical_probs, 1 - categorical_probs), axis=1
            )

            # Posterior probability
            probs[:, i] = numeric_likelihood * cat_likelihood * prior

        # Normalize to get probabilities
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs /= np.where(probs_sum == 0, 1e-9, probs_sum)  # Handle division by zero
        return probs

    def predict(self, X):
        """
        Predict class labels (classification) or regression targets (regression).

        Parameters:
        - X: array-like of shape (n_samples, n_features), input data.

        Returns:
        - labels: array-like of shape (n_samples,), predicted class labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure all categorical columns are one-hot encoded
        for col in self.categorical_cols:
            unique_values = np.unique(X[:, col])
            if not (set(unique_values) <= set([0, 1]) or set(unique_values) <= set([0., 1.])):
                raise ValueError(f"Categorical column at index {col} is not one-hot encoded.")

        if self.is_classification:
            probs = self.predict_proba(X)
            predictions = np.argmax(probs, axis=1)
            return self.label_encoder_.inverse_transform(predictions)
            
        else:
            # # Compute posterior probabilities for numeric features
            # numeric_probs = self.numeric_gmm_.predict_proba(X[:, self.numeric_cols])

            # # Incorporate one-hot categorical probabilities
            # categorical_probs = np.prod(np.where(X[:, self.categorical_cols] == 1, 
            #                                     self.categorical_probs_, 
            #                                     1 - self.categorical_probs_), axis=1)

            # # Combine numeric and categorical probabilities
            # combined_probs = numeric_probs * categorical_probs[:, np.newaxis]

            # # Compute expected value of y given the combined probabilities
            # target_means = self.target_gmm_.means_.flatten()  # Means of y for each component
            # predictions = np.dot(combined_probs, target_means)

            # return predictions

            n_samples = X.shape[0]
            n_components = self.target_gmm_.means_.shape[0]  # Use actual components count
            probs = np.zeros((n_samples, n_components))

            for component_num in range(n_components):
                if component_num not in self.models_:
                    continue  # Skip components that had no training samples

                model = self.models_[component_num]
                gmm = model['gmm']
                categorical_probs = model['categorical_probs']

                # Compute numeric likelihood if applicable
                if self.numeric_cols and gmm is not None:
                    log_probs = gmm.score_samples(X[:, self.numeric_cols])
                    numeric_likelihood = np.exp(log_probs - np.max(log_probs))
                else:
                    numeric_likelihood = np.ones(n_samples)

                # Compute categorical likelihood
                cat_likelihood = np.prod(
                    np.where(X[:, self.categorical_cols] == 1,
                            categorical_probs,
                            1 - categorical_probs),
                    axis=1
                )

                # Combine probabilities with component weight
                probs[:, component_num] = (
                    numeric_likelihood * 
                    cat_likelihood * 
                    self.target_gmm_.weights_[component_num]
                )

            # Normalize probabilities across components
            probs = probs / probs.sum(axis=1, keepdims=True)
            
            # Calculate weighted average using target means
            target_means = self.target_gmm_.means_.flatten()
            predictions = np.dot(probs, target_means)
            
            return predictions
                    
            




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
