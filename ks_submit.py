import argparse
import ctf4science
parser = argparse.ArgumentParser(
                    prog='submitks',
                    )

parser.add_argument('--pair_id', type=int)
args = parser.parse_args()
pair_id = args.pair_id

prediction_length = 1000

if pair_id == 1:
    arr, _ = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X1train'
elif pair_id in [2,3]:
    arr, _ = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X2train'
elif pair_id in [4,5]:
    arr, _ = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X3train'
elif pair_id == 6:
    arr, _ = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X4train'
elif pair_id == 7:
    arr, _ = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X5train'
elif pair_id == 8:
    _, arr = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X9train'
elif pair_id == 9:
    _, arr = ctf4science.data_module.load_dataset('KS_Official', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X10train'
else:
    raise ValueError('Incorrect pair_id')


from datasets import load_dataset
from tabpfn_time_series import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import to_gluonts_univariate, generate_test_X
import pandas as pd


import numpy as np
import pandas as pd
from tabpfn_time_series import TimeSeriesDataFrame
from scipy.io import loadmat

# Example input: (timesteps=3, n_items=3)
# arr = loadmat('data/KS_Official/train/' + load_string + '.mat')[load_string][-500:]
if pair_id not in [2,4]:
    arr = np.vstack([arr, np.zeros((1000, 1024))])
else:
    arr = loadmat('data/KS_Official/train/' + load_string + '.mat')[load_string]
timesteps, n_items = arr.shape

# Generate timestamps (e.g., daily starting from 2019-01-01)
start_date = pd.Timestamp("2019-01-01")
timestamps = pd.date_range(start=start_date, periods=timesteps, freq='T')

# Create multi-index: one level for item_id, one for timestamp
multi_index = pd.MultiIndex.from_product(
    [range(n_items), timestamps],
    names=["item_id", "timestamp"]
)

# Flatten array in column-major order (i.e., per series)
flattened = arr.T.flatten()  # shape: (n_items * timesteps,)

# Build the DataFrame
df = pd.DataFrame({"target": flattened}, index=multi_index)

# Create TimeSeriesDataFrame
tsdf = TimeSeriesDataFrame(df)

print(tsdf.head())


tsdf = tsdf[
    tsdf.index.get_level_values("item_id").isin(tsdf.item_ids[:1024])
]
if pair_id not in [2,4]:
    train_tsdf, test_tsdf_ground_truth = tsdf.train_test_split(
        prediction_length=prediction_length
    )
    test_tsdf = generate_test_X(train_tsdf, prediction_length)

else:
    train_tsdf = tsdf
    test_tsdf = generate_test_X(train_tsdf, prediction_length)

from tabpfn_time_series import FeatureTransformer
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

selected_features = [
    RunningIndexFeature(),
    CalendarFeature(),
    AutoSeasonalFeature(),
]

feature_transformer = FeatureTransformer(selected_features)

train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)

from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

predictor = TabPFNTimeSeriesPredictor(
    tabpfn_mode=TabPFNMode.LOCAL,
)

if pair_id not in [2,4]:
    pred = predictor.predict(train_tsdf, test_tsdf)

else:
    print('2 or 4')
    pred = predictor.predict(train_tsdf, train_tsdf)
temp_list = []
for i in range(1024):
    temp_list.append(pred['target'][i].to_numpy())

pred_arr = np.array(temp_list).T
np.savez('pairid' + str(pair_id) + 'ks.npz', data_mat = pred_arr)
