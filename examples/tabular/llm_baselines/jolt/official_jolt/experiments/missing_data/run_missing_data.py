import os
from run_jolt import run_jolt
from hf_api import get_model_and_tokenizer, llm_map
from parse_args import init_option_parser
from jsonargparse import ArgumentParser


seeds = ['0', '1', '2']
shots = ['10', '20', '30', '40', '50']
missings = ['0.1', '0.2', '0.3', '0.4', '0.5']
test_size = 200

def main():
    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--llm_path', type=str, help='Path to LLM.')
    parser.add_argument("--llm_type", choices=llm_map.keys(), help="Hugging face model to use.")
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()
    print(args, flush=True)

    option_parser = init_option_parser()

    model, tokenizer = get_model_and_tokenizer(args)
    for shot in shots:
        for seed in seeds:
            for missing in missings:
                print("-------------------------------")
                print("model={}, dataset=wine_quality, shot={}, seed={}, missing={}".format(args.llm_type, shot, seed, missing))
                print("-------------------------------")
                config_args = option_parser.parse_args(args=[
                        "--experiment_name", "wine_quality_shot_{}_seed_{}_missing_{}_{}".format(shot, seed, missing, args.llm_type),
                        "--data_path", "./data/wine_quality.csv",
                        "--output_dir", "./experiments/missing_data/output",
                        "--mode", "sample_logpy",
                        "--seed", seed,
                        "--y_column_types", "numerical", "categorical",
                        "--batch_size", str(args.batch_size),
                        "--num_samples", str(10),
                        "--num_decimal_places_x", str(3),
                        "--num_decimal_places_y", str(1),
                        "--y_column_names", "alcohol", "quality",
                        "--max_generated_length", str(120),
                        "--train_size_limit", shot,
                        "--test_size_limit", str(test_size),
                        "--header_option", "headers_as_item_prefix",
                        "--prefix", "The data contains features that determine the quality of wine. Predict the alcohol content and the quality score of each wine based on the features.\n",
                        "--csv_split_option", "fixed_train_and_test_size",
                        "--missing_fraction", missing,
                ])
                run_jolt(args=config_args, model=model, tokenizer=tokenizer)

    for shot in shots:
        for seed in seeds:
            for missing in missings:
                print("-------------------------------")
                print("model={}, dataset=car, shot={}, seed={}, missing={}".format(args.llm_type, shot, seed, missing))
                print("-------------------------------")
                config_args = option_parser.parse_args(args=[
                        "--experiment_name", "car_shot_{}_seed_{}_missing_{}_{}".format(shot, seed, missing, args.llm_type),
                        "--data_path", "./data/car_evaluation.csv",
                        "--output_dir", "./experiments/missing_data/output",
                        "--mode", "logpy_only",
                        "--seed", seed,
                        "--y_column_types", "categorical",
                        "--batch_size", str(args.batch_size),
                        "--num_decimal_places_x", str(0),
                        "--num_decimal_places_y", str(0),
                        "--y_column_names", "class",
                        "--max_generated_length", str(50),
                        "--train_size_limit", shot,
                        "--test_size_limit", str(test_size),
                        "--header_option", "headers_as_item_prefix",
                        "--prefix", "The data contains features that describe a car. Predict the overall evaluation class (unacceptable, acceptable, good, or very good).\n",
                        "--csv_split_option", "fixed_train_and_test_size",
                        "--missing_fraction", missing,
                ])
                run_jolt(args=config_args, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
