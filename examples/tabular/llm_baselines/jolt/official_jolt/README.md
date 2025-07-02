
# Code for JoLT: Joint Probabilistic Predictions on Tabular Data Using LLMs
This repository contains the code to reproduce the experiments carried out in [JoLT: Joint Probabilistic Predictions on Tabular Data Using LLMs](https://arxiv.org/pdf/2502.11877).

The code has been authored by: John Bronskill, Aliaksandra Shysheya, Shoaib Ahmed Siddiqui, and James Requeima.

## Dependencies
This code requires the following:
* python 3.9 or greater
* PyTorch 2.6.0 or greater
* transformers 4.48.3 or greater
* accelerate 1.3.0 or greater
* jsonargparse 4.36.0 or greater
* numpy 2.0.2 or greater
* scikit-learn 1.6.1 or greater
* scipy 1.13.1 or greater
* pandas 2.2.3 or greater

## LLM Support and GPU Requirements
We support a variety of LLMs through the Hugging Face transformer APIs. The code currently supports the following
LLMs:

| LLM Type     | URL    | GPU Memory Required (GB) |
| ---      | ---    |--------------------------|
| deepseek-r1-distill-qwen-7B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 24                    |
| gemma-2-2B | https://huggingface.co/google/gemma-2-2b | 8                    |
| gemma-2-2B-instruct | https://huggingface.co/google/gemma-2-2b-it | 8                    |
| gemma-2-9B | https://huggingface.co/google/gemma-2-9b | 24                    |
| gemma-2-9B-instruct | https://huggingface.co/google/gemma-2-9b-it | 24                    |
| gemma-2-27B | https://huggingface.co/google/gemma-2-27b | 80                    |
| gemma-2-27B-instruct | https://huggingface.co/google/gemma-2-27b-it | 80                    |
| llama-2-7B | https://huggingface.co/meta-llama/Llama-2-7b | 24                    |
| llama-2-70B | https://huggingface.co/meta-llama/Llama-2-70b | 160                   |
| llama-3-8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B | 24                    |
| llama-3-70B | https://huggingface.co/meta-llama/Meta-Llama-3-70B | 160                   |
| llama-3.1-70B-instruct | https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct | 160                   |
| mixtral-8x7B | https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 | 24                    |
| mixtral-8x7B-instruct | https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 | 160                   |
| phi-3-mini-128k-instruct | https://huggingface.co/microsoft/Phi-3-mini-128k-instruct | 8                     |
| phi-3-small-128k-instruct | https://huggingface.co/microsoft/Phi-3-small-128k-instruct | 24                     |
| phi-3.5-mini-instruct | https://huggingface.co/microsoft/Phi-3.5-mini-instruct | 8                     |
| phi-3.5-moe-instruct | https://huggingface.co/microsoft/Phi-3.5-MoE-instruct | 160                     |
| qwen2.5-1.5B-instruct | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | 8                     |
| qwen2.5-7B-instruct | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | 24                     |
| qwen2.5-72B-instruct | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct | 160                     |

Adding a new LLM that supports the hugging face APIs is not difficult, just modify ```hf_api.py```.

## Installation
1. Clone or download this repository.
2. Install the python libraries listed under dependencies.

## Running the code
* Change directory to the root directory of this repo.
* On linux run:
  * ```export PYTHONPATH=.```
* On Windows run:
  * ```set PYTHONPATH=.```

## Reproducing the Experiments
From the root directory of the repo, run any of the commands below.
### Classification Setting
```python ./experiments/classification/run_classification.py --llm_type <LLM Type> --batch_size <value>```

### Multi-target Prediction
#### Wine Quality
```python ./experiments/multi_target_prediction/run_wine_quality.py --llm_type <LLM Type> --batch_size <value>```
#### Medals
```python run_jolt.py --experiment_name medals --data_path data/medals.csv --llm_type <LLM Type> --output_dir experiments/multi_target_prediction/output --num_samples 1 --batch_size 5 --mode sample_logpy --num_decimal_places_x 0 --num_decimal_places_y 0 --y_column_types numerical numerical --y_column_names 'Silver Medal Count' 'Gold Medal Count' --max_generated_length 25 --header_option headers_as_item_prefix --top_k 1 --prefix 'Each example contains five columns: Olympic Year, Country, Bronze Medal Count, Silver Medal Count, and Gold Medal Count that describe what type and how many medals a country won at the Olympic games that year.  Predict the number of silver and gold medals won by that country in that year.\n' --columns_to_ignore Country_Label --csv_split_option fixed_indices --train_start_index 10 --train_end_index 80 --test_start_index 0 --test_end_index 10```
#### Movies
```python run_jolt.py --experiment_name movies --data_path data/movies.csv --llm_type <LLM Type> --output_dir experiments/multi_target_prediction/output --num_samples 1 --batch_size 13 --mode sample_logpy --num_decimal_places_x 1 --num_decimal_places_y 1 --y_column_types numerical categorical categorical categorical categorical categorical categorical categorical categorical --y_column_names Rating Adventure Comedy Family Action Fantasy Thriller Drama Horror --max_generated_length 50 --header_option headers_as_item_prefix --top_k 1 --prefix 'Each example contains 11 columns: Movie Name, Revenue in Millions of Dollars, Rating, and 8 genre tags (Adventure, Comedy, Family, Action, Fantasy, Thriller, Drama, and Horror). Predict the movie rating and genre tags.' --columns_to_ignore 'Release Date' Animation 'Science Fiction' Crime Romance Music History Mystery --csv_split_option fixed_indices --train_start_index 0 --train_end_index 89 --test_start_index 89 --test_end_index 188```
### Handling Missing Data
```python ./experiments/missing_data/run_missing_data.py --llm_type <LLM Type> --batch_size <value>```
### Missing Data Imputation
```python ./experiments/imputation/run_imputation.py --experiment_name medals_imputation --data_path data/paris_2024_medals.csv --llm_type <LLM Type> --output_dir ./experiments/imputation/output --num_samples 10 --batch_size 5 --mode sample_logpy --num_decimal_places_x 0 --num_decimal_places_y 0 --max_generated_length 40 --header_option headers_as_item_prefix --seed 0 --missing_fraction 0.2  --impute_features True --prefix 'Each row of data contains the name of a country and how many gold, silver and bronze medals that country won at the Paris 2024 olympics.'```
### Custom Datasets
You can use your own datasets. The default train/test split is set to 80%/20%, but can be customized. See the CSV input options in```parse_args.py```.

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our paper:
```
@misc{shysheya2025joltjointprobabilisticpredictions,
      title={JoLT: Joint Probabilistic Predictions on Tabular Data Using LLMs}, 
      author={Aliaksandra Shysheya and John Bronskill and James Requeima and Shoaib Ahmed Siddiqui and Javier González and David Duvenaud and Richard E. Turner},
      year={2025},
      eprint={2502.11877},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2502.11877}, 
}
```
