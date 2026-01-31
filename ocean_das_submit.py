import argparse
import ctf4science
import ctf4science.data_module

parser = argparse.ArgumentParser(
                    prog='submitoceandas',
                    )

parser.add_argument('--pair_id', type=int)
args = parser.parse_args()
pair_id = args.pair_id

prediction_length = 1000
spatial_dim = 3000

# ocean_das pair mappings based on ocean_das.yaml:
# Pair 1: X1train.npz
# Pair 2: X2train.npz (reconstruction)
# Pair 3: X2train.npz
# Pair 4: X3train.npz (reconstruction)
# Pair 5: X3train.npz
# Pair 6: X4train.npz
# Pair 7: X5train.npz
# Pair 8: X6train+X7train+X8train (initialization: X9train)
# Pair 9: X6train+X7train+X8train (initialization: X10train)

if pair_id == 1:
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X1train'
elif pair_id in [2, 3]:
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X2train'
elif pair_id in [4, 5]:
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X3train'
elif pair_id == 6:
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X4train'
elif pair_id == 7:
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]
    load_string = 'X5train'
elif pair_id == 8:
    # Multi-file training with initialization
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]  # Uses combined X6train, X7train, X8train
    load_string = 'X6train'
elif pair_id == 9:
    # Multi-file training with initialization
    arr, _ = ctf4science.data_module.load_dataset('ocean_das', pair_id, transpose=False)
    arr = arr[0]  # Uses combined X6train, X7train, X8train
    load_string = 'X6train'
else:
    raise ValueError('Incorrect pair_id')


from datasets import load_dataset
from tabpfn_time_series import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import to_gluonts_univariate, generate_test_X
import pandas as pd


import numpy as np
import pandas as pd
from tabpfn_time_series import TimeSeriesDataFrame

# For ocean_das: pairs 2 and 4 are reconstruction tasks
if pair_id not in [2, 4]:
    arr = np.vstack([arr, np.zeros((1000, spatial_dim))])
else:
    # For reconstruction tasks, load the full training data
    data = np.load('data/ocean_das/train/' + load_string + '.npz')
    arr = data['data_mat'] if 'data_mat' in data else data[list(data.keys())[0]]
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
    tsdf.index.get_level_values("item_id").isin(tsdf.item_ids[:spatial_dim])
]
if pair_id not in [2, 4]:
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

if pair_id not in [2, 4]:
    pred = predictor.predict(train_tsdf, test_tsdf)

else:
    print('2 or 4')
    pred = predictor.predict(train_tsdf, train_tsdf)
temp_list = []
for i in range(spatial_dim):
    temp_list.append(pred['target'][i].to_numpy())

pred_arr = np.array(temp_list).T
np.savez('pairid' + str(pair_id) + 'ocean_das.npz', data_mat=pred_arr)
