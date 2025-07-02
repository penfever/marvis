import os
from run_jolt import run_jolt
from hf_api import get_model_and_tokenizer, llm_map
from parse_args import init_option_parser
from jsonargparse import ArgumentParser


seeds = ['0', '1', '2', '3', '4']
shots = ['10', '20', '30', '40', '50']

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
            print("-------------------------------")
            print("model={}, dataset=wine_quality, shot={}, seed={}".format(args.llm_type, shot, seed))
            print("-------------------------------")
            config_args = option_parser.parse_args(args=[
                    "--experiment_name", "wine_quality_shot_{}_seed_{}_{}".format(shot, seed, args.llm_type),
                    "--data_path", "./data/wine_quality.csv",
                    "--output_dir", "./experiments/multi_target_prediction/output",
                    "--mode", "sample_logpy",
                    "--seed", seed,
                    "--y_column_types", "numerical", "categorical",
                    "--batch_size", str(args.batch_size),
                    "--num_samples", str(1),
                    "--num_decimal_places_x", str(3),
                    "--num_decimal_places_y", str(1),
                    "--y_column_names", "alcohol", "quality",
                    "--max_generated_length", str(20),
                    "--train_size_limit", shot,
                    "--test_size_limit", str(200),
                    "--header_option", "headers_as_item_prefix",
                    "--top_k",  str(1),
                    "--prefix", "The data contains features that determine the quality of wine. Predict the alcohol content and the quality score of each wine based on the features.\n",
                    "--csv_split_option", "fixed_train_and_test_size"
            ])
            run_jolt(args=config_args, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()

