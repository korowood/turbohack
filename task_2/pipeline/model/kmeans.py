from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from typing import List
import pandas
import numpy
import numpy.matlib
from pipeline import Dataset
from pipeline.metrics import v_measure_parallel


class ParallelKMeans:
    def __init__(
            self,
            init: str = 'k-means++',
            n_init: int = 3,
            max_iter: int = 50,
            tol: float = 1e-3,
            verbose: int = 0,
            random_state: int = 42,
            copy_x: bool = True,
            algorithm: str = 'lloyd'
    ):
        self.models = dict()
        self.single_model = None
        self.single_predict = None
        self.predictions = None
        self.optim_score = None
        self.inertia = None

        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _kmeans(self, data: pandas.DataFrame, k: int, feature: str = None):
        model = KMeans(
            n_clusters=k,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm
        )
        pred = model.fit_predict(data.values)
        return [feature, pred, model]

    def fit_predict(
            self,
            data: Dataset,
            features: List[str] = None,
            k: int = 10,
            save: bool = True
    ):
        if features is None:
            features = data.features
        _, predict, model = self._kmeans(data=data.data[features], k=k)
        predict = pandas.Series(data=predict, index=data.data.index, name='predict')
        if save:
            self.single_predict = predict
            self.single_model = model
        return predict

    def iterfit(
            self,
            data: Dataset,
            features: List[str] = None,
            k: int = 10,
            save: bool = True,
            tqdm_on: bool = True,
            n_jobs: int = -1
    ) -> pandas.DataFrame:
        if features is None:
            features = data.features
        iterator = features
        if tqdm_on:
            iterator = tqdm(iterator)
        result = Parallel(n_jobs=n_jobs)(
            delayed(self._kmeans)(
                feature=feature,
                data=data.data[[feature]], k=k
            ) for feature in iterator
        )
        predictions_dict = dict()
        for feature, pred, model in result:
            predictions_dict[feature] = pred
            if save:
                self.models[feature] = model
        predictions = pandas.DataFrame({feat: predictions_dict[feat] for feat in features})
        if save:
            self.predictions = predictions
        return predictions

    def optimize(self, data: Dataset, features: List[str], k_list: List[int] = None, tqdm_on: bool = True) -> int:
        if features is None:
            features = data.features
        if k_list is None:
            k_list = list(range(2, 40))
        iterator = k_list
        if tqdm_on:
            iterator = tqdm(iterator)
        optim_score = {'k': list(), 'v_score': list()}
        for k in iterator:
            preds = self.iterfit(data, features=features, k=k, tqdm_on=False, save=False)
            v_score = v_measure_parallel(preds, n_jobs=-1, tqdm_on=False)
            optim_score['k'].append(k)
            optim_score['v_score'].append(v_score.values.mean())
        self.optim_score = pandas.DataFrame(optim_score)
        best_k = self.optim_score.iloc[self.optim_score['v_score'].idxmax()]['k']
        return best_k

    def elbow(self, data: Dataset, features: List[str], k_list: List[int] = None, tqdm_on: bool = True, n_jobs: int = -1):
        if features is None:
            features = data.features
        if k_list is None:
            k_list = list(range(2, 40))
        iterator = k_list
        if tqdm_on:
            iterator = tqdm(iterator)

        result = Parallel(n_jobs=n_jobs)(
            delayed(lambda data, k: [k, self._kmeans(data=data, k=k)])(
                data=data.data[features],
                k=k
            ) for k in iterator
        )
        result = sorted(result, key=lambda x: x[0])
        elbow = {'k': list(), 'inertia': list()}
        for k, res in result:
            elbow['k'].append(k)
            elbow['inertia'].append(res[2].inertia_)
        self.inertia = pandas.DataFrame(elbow)

        curve = self.inertia['inertia'].to_numpy()
        n_points = len(iterator)
        all_coord = numpy.vstack((range(n_points), curve)).T
        numpy.array([range(n_points), curve])
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / numpy.sqrt(numpy.sum(line_vec ** 2))
        vec_from_first = all_coord - first_point
        scalar_product = numpy.sum(vec_from_first * numpy.matlib.repmat(line_vec_norm, n_points, 1), axis=1)
        vec_from_first_parallel = numpy.outer(scalar_product, line_vec_norm)
        vec_t_line = vec_from_first - vec_from_first_parallel
        dist_to_line = numpy.sqrt(numpy.sum(vec_t_line ** 2, axis=1))
        idx = numpy.argmax(dist_to_line)
        best_k = self.inertia['k'].iloc[idx]
        return best_k

    def plot(self, data: Dataset, features: List[str], k: int = 10):
        if features is None:
            features = data.features
        pca = PCA(2)
        df_pca = pca.fit_transform(data.data[features])
        _, _, model = self._kmeans(data=data.data[features], k=k)
        kmeans_labels = model.labels_

        data = pandas.DataFrame()
        data['label'] = kmeans_labels

        #kmeans_centers = model.cluster_centers_

        #centr_pca = pca.transform(kmeans_centers)

        data['pca-one'] = df_pca[:, 0]
        data['pca-two'] = df_pca[:, 1]

        plt.figure(figsize=(16, 10))
        seaborn.scatterplot(
            x="pca-one", y="pca-two",
            palette=seaborn.color_palette("hls", len(set(kmeans_labels))),
            data=data,
            hue='label',
            legend="full",
            alpha=0.5
        )

