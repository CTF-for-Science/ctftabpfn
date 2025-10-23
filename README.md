### tapfn_time_series for ctf4science

This submodule contains code to evaluate the foundation model TabPFNv2 (https://github.com/PriorLabs/tabpfn-time-series) on ct4science benchmarks. config_KS.yaml is a configuration file to run on all KS_Official tasks and config_Lorenz is the same for Lorenz_Official. 

## Usage
First, ensure that `ctf4science` is already installed. Then, from the root directory of `ctftabpfn`, 
```bash
pip install -r requirements.txt
```
will install the remaining dependencies. 

Given a config file, one can train and evaluate a model by running
```bash
python run.py <path-to-config>
```
To generate a prediction matrix for new dataset and pair_id, follow the examples in `ks_submit.py` or `lorenz_submit.py`. For instance, running
```bash
python lorenz_submit.py --pair_id 1
```
will generate the prediction matrix for pair_id 1 for the Lorenz_Official data. 

Citation for TabPFNv2:

@misc{hoo2025tablestimetabpfnv2outperforms,
      title={From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models}, 
      author={Shi Bin Hoo and Samuel Müller and David Salinas and Frank Hutter},
      year={2025},
      eprint={2501.02945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.02945}, 
}