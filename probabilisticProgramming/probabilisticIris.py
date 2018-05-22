import numpy as np
import pymc3 as pm
from sklearn.datasets import load_iris

# data processing
dataset = load_iris()
target = dataset['target']
data = dataset['data'] #[sepal ]
target_names = dataset['target_names']
data_names = dataset['feature_names']
Y1 = data[:50]
Y2 = data[50:100]
Y3 = data[100:]

with pm.Model() as modelSetosa:
    sepal_length = np.array([i[0] for i in Y1])
    sepal_width = np.array([i[1] for i in Y1])
    petal_length = np.array([i[2] for i in Y1])
    petal_width = np.array([i[3] for i in Y1])

    intercept = pm.Normal('intercept',mu=0, sd=10)
    coefficients = pm.Normal('coefficients', mu=0, sd=10, shape=4)
    sd = pm.Normal('sd', sd=1)
    expected = intercept + coefficients[0]*sepal_length + coefficients[1]*sepal_width + coefficients[2]*petal_length + coefficients[3]*petal_width
    print(expected.shape[0])
    likelihood = pm.Normal('likelihood', mu=expected, sd=sd, observed=Y1)

estimate = pm.find_MAP(model=modelSetosa)
print(estimate)