#%%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from scipy.spatial import KDTree
import numpy as np

# %%
"""
Say you use two images before and two after the target, and 3 bands per image
These images will have to be selected based on whether they have valid pixels where the target is missing
Actually not true. We need to select similar pixels where valid data exists in the target image.
It's the valid pixels in the target image that we'll use to fill missing pixels.
Do all images will need to be loaded into memory at once?
"""
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh = KNeighborsRegressor(n_neighbors=2).fit(X, y)

# %%
neigh.predict([[1.5]])
# %%

data = np.array(
    [[1.0, np.nan, 1.0], [1.1, 2.0, 1.0], [4.0, 2.1, 3.0], [10.0, 11.0, 12.0]]
)

tree = KDTree(
    data,
    leafsize=10,
    compact_nodes=True,
    copy_data=False,
    balanced_tree=True,
    boxsize=None,
)
tree.query([0.9, np.nan, 0.9])  # Doesn't work!
# %%
imputer = KNNImputer(n_neighbors=2, weights="distance").fit(data)
imputer.transform(np.array([[1.0, np.nan, 1.0]]))
# TODO: sample for pixels that have the mostly complete timeseries
# maybe stratify the sample based on landcover
# Fit the transform to the sampled data.
# Load in chunks of the images and call transform on them and write out the chunks!
# %%
