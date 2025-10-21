
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
import os

from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization


def main(config_path: str) -> None:
    """
    Main function to run the naive baseline model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"

    # Generate a unique batch_id for this run, you can add any descriptions you want
    #   e.g. f"batch_{learning_rate}_"
    batch_id = f"batch_"
    # temp = str(config['model']['Wr_spectral_radius'])
    # batch_id = batch_id + temp
 
    # Define the name of the output folder for your batch
    batch_id = f"{batch_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        
        if dataset_name == 'Lorenz_Official':
            os.system('python lorenz_submit.py --pair_id ' + str(pair_id))
        elif dataset_name == 'KS_Official':
            os.system('python ks_submit.py --pair_id ' + str(pair_id))
        else:
            pass

        if dataset_name == 'Lorenz_Official':
            pred_data = np.load('pairid' + str(pair_id) + 'lorenz.npz')['data_mat'].T
        elif dataset_name == 'KS_Official':
            pred_data = np.load('pairid' + str(pair_id) + 'ks.npz')['data_mat'].T
        else: 
            pass

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data.T)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data.T, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
