import time 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor 
import pandas as pd  


df_train = pd.read_csv('../dataset/feature/train_feature/car12_features.csv') 
#df_train = pd.read_csv('../dataset/feature/energy_train_1029_features.csv') 

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers 

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
#    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))] 

energy_dsoc = df_train[['dsoc', 'charge_energy']].values
energy_hour = df_train[['charge_hour', 'charge_energy']].values
dsoc_hour = df_train[['charge_hour', 'dsoc']].values
datasets = []
datasets.append(energy_dsoc)
datasets.append(energy_hour)
datasets.append(dsoc_hour)

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):  
    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # plot the levels lines and the points
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])
 
        plt.xticks(())
        plt.yticks(()) 
        plot_num += 1

plt.show()