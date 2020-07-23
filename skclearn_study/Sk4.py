
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale
import numpy as np

onehot_encoder = DictVectorizer()

X = [
    {'city':'new york'},
    {'city':'san francisco'},
    {'city' : 'chapel hill'}
]

print(onehot_encoder.fit_transform(X))

X = np.array([
    [0.,0.,5.,13.,9.,1.],
    [0.,0.,13.,15.,10.,15.],
    [0.,3.,15.,2.,0.,11.]
])

print(scale(X))