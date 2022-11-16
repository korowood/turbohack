from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pandas
import numpy
from sklearn.metrics.cluster import v_measure_score


def v_measure_parallel(data: pandas.DataFrame, n_jobs: int = -1, tqdm_on: bool = True) -> pandas.DataFrame:
    def metric(data: pandas.DataFrame, idx: int):
        pred1 = data[data.columns[idx]]
        metrics = list()
        for idx2 in range(len(data.columns)):
            if idx2 < idx:
                metrics.append(0)
            elif idx2 == idx:
                metrics.append(1)
            else:
                pred2 = data[data.columns[idx2]]
                metrics.append(v_measure_score(pred1, pred2))
        return [data.columns[idx], metrics]
    # if features is None:
    #     features = [col for col in data.columns if col not in ['date_col', 'target']]
    iterator = list(range(len(data.columns)))
    if tqdm_on:
        iterator = tqdm(iterator)
    metrics = Parallel(n_jobs=n_jobs)(delayed(metric)(data, idx) for idx in iterator)
    metrics_dict = {feature: value for feature, value in metrics}
    result = list()
    for feat in data.columns:
        result.append(metrics_dict[feat])
    result = numpy.array(result)
    result = result + result.T
    result_pdf = pandas.DataFrame(data=result, columns=data.columns, index=data.columns)
    return result_pdf
