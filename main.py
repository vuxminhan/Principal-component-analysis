import numpy as np
from time import time
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class ModularPCA:
    def __init__(self, n_components=None, svd_solver='auto', whiten=False, random_state=None, num_regions=4):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.num_regions = num_regions
        self.pca_models = []

    def fit(self, X, y=None):
        # Split each face image into regions
        X_regions = np.split(X, self.num_regions, axis=1)

        # Apply PCA separately to each region
        self.pca_models = []
        for region in X_regions:
            pca = PCA(
                n_components=self.n_components,
                svd_solver=self.svd_solver,
                whiten=self.whiten,
                random_state=self.random_state
            )
            pca.fit(region)
            self.pca_models.append(pca)

        return self

    def transform(self, X):
        # Transform each region using its corresponding PCA model
        X_transformed = np.hstack([pca.transform(region) for pca, region in zip(self.pca_models, np.split(X, self.num_regions, axis=1))])
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
class FaceRecognitionModel:
    def __init__(self, classifier, pca_method, data_path='/content/drive/MyDrive/Colab Notebooks/ATT_images', n_components=50, verbose = False,
                 tunning = False, param_grid = None):
        self.data_path = data_path
        self.n_components = n_components
        self.images = []
        self.labels = []
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.pca = pca_method
        self.clf = classifier
        self.verbose = verbose
        self.tunning = tunning
        self.param_grid = param_grid
        self.y_pred = None

    def load_data(self):
        for i in range(1, 41):
            for j in range(1, 11):
                file_path = os.path.join(self.data_path,'s'+str(i),str(j)+'.pgm')
                img = cv2.imread(file_path, 0)
                img = img.flatten()
                self.images.append(img)
                self.labels.append(i)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def split_data(self):
        x_train, x_test, y_train, y_test = [], [], [], []
        for i in range(1, 41):
            indices = np.where(self.labels == i)[0]
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=1234)

            x_train.extend(self.images[train_indices])
            y_train.extend(self.labels[train_indices])

            x_test.extend(self.images[test_indices])
            y_test.extend(self.labels[test_indices])

        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_test, self.y_test = np.array(x_test), np.array(y_test).reshape(-1,1)

        # ### Shuffle
        # merged_test = np.hstack((x_test, y_test))
        # shuffled_test = np.random.permutation(merged_test)
        # self.x_test, self.y_test = shuffled_test[:,:-1], shuffled_test[:,-1]

    def perform_pca(self):
        print(f"\tExtracting the top {self.n_components} eigenfaces from {self.x_train.shape[0]} faces")
        t0 = time()
        self.pca = self.pca.fit(self.x_train)
        print("\tDone in %0.3fs" % (time() - t0))

    def project_on_eigenfaces(self):
        print("\tProjecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        self.x_train = self.pca.transform(self.x_train)
        self.x_test = self.pca.transform(self.x_test)
        print("\tDone in %0.3fs" % (time() - t0))

    def train_classifier(self):
        print("\tFitting the classifier to the training set")
        t0 = time()
        self.clf.fit(self.x_train, self.y_train)
        print("\tDone in %0.3fs" % (time() - t0))

    def print_classifier_info(self):
        print("\tClassifier:", self.clf)
        print("\tHyperparameters:", self.clf.get_params())

    def evaluate_model(self):
        print("\tPredicting the people names on the testing set")
        t0 = time()
        self.y_pred = self.clf.predict(self.x_test)

        print("\tDone in %0.3fs" % (time() - t0))
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f_score = f1_score(self.y_test, self.y_pred, average='weighted')

        # Calculate AUC
        auc_per_class = []

        for i in range(1,41):
            # Convert the problem into a binary classification problem for each class
            binary_true = (self.y_test == i).astype(int)
            binary_predict = (self.y_pred == i).astype(int)

            # Calculate AUC for each class
            auc_class = roc_auc_score(binary_true, binary_predict)
            auc_per_class.append(auc_class)

        # Calculate the average AUC
        auc_score = np.mean(auc_per_class)

        recognition_rate = (float(accuracy) +float(f_score) + float(auc_score))/3

        print("\n\tClassification Report:")
        print(f'\t\tAccuracy: {accuracy}\n')
        print(f'\t\tF1_Score: {f_score}\n')
        print(f'\t\tAUC_Score: {auc_score}\n')
        print(f'\t\tRecognition Rate: {recognition_rate}\n')

    def visualize_error(self):
      incorrect_indices = np.where(self.y_pred != self.y_test.flatten())[0]
      if len(incorrect_indices) > 0:
            print("\n\tVisualizing images with wrong labels:")
            plt.figure(figsize=(16, 9))
            for i, idx in enumerate(incorrect_indices[:min(25, len(incorrect_indices))]):
                plt.subplot(5, 5, i + 1)
                img = self.x_test[idx]
                img = self.pca.inverse_transform(img)  # Transform back to original space
                img = img.reshape((112, 92))
                plt.imshow(img, cmap='gray')
                plt.title(f'True: {int(self.y_test[idx])}, Predicted: {int(self.y_pred[idx])}')
                plt.xticks(())
                plt.yticks(())
            plt.show()


    def plot_eigenfaces(self):
        eigenfaces = self.pca.components_.reshape((self.n_components, 112, 92))

        plt.figure(figsize=(16, 9))
        for i in range(self.n_components):
            plt.subplot(10, 10, i + 1)
            plt.imshow(eigenfaces[i], cmap='gray')
            plt.xticks(())
            plt.yticks(())
        plt.show()

    def hyperparameter_tuning(self):
      print("\tPerforming hyperparameter tuning with GridSearchCV")
      t0 = time()

      grid_search = GridSearchCV(self.clf, param_grid=self.param_grid, cv=5, n_jobs=-1)
      grid_search.fit(self.x_train, self.y_train)

      self.clf = grid_search.best_estimator_
      print("\tBest hyperparameters found: ", grid_search.best_params_)
      print("\tDone in %0.3fs" % (time() - t0))

    def execute(self):
      self.load_data()
      self.split_data()
      self.perform_pca()
      self.project_on_eigenfaces()

      if self.tunning:
        self.hyperparameter_tuning()

      self.train_classifier()
      self.evaluate_model()

      if self.verbose:
        self.print_classifier_info()
        self.plot_eigenfaces()

