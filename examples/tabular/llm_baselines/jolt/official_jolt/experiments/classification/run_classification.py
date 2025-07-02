import os
from run_jolt import run_jolt
from hf_api import get_model_and_tokenizer, llm_map
from parse_args import init_option_parser
from jsonargparse import ArgumentParser


seeds = ['0', '1', '2', '3', '4']
shots = ['0', '4', '8', '16', '32']
datasets = ['bank', 'blood', 'calhousing', 'car', 'creditg', 'diabetes', 'heart', 'income', 'jungle']

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
    for dataset in datasets:
        for shot in shots:
            for seed in seeds:
                print("-------------------------------")
                print("model={}, dataset={}, shot={}, seed={}".format(args.llm_type, dataset, shot, seed))
                print("-------------------------------")
                config_args = option_parser.parse_args(args=[
                        "--experiment_name", "{}_shot_{}_seed_{}_{}".format(dataset, shot, seed, args.llm_type),
                        "--data_path", os.path.join("./data/tabllm", "tabllm_{}_{}_{}.pkl".format(dataset, shot, seed)),
                        "--output_dir", "./experiments/classification/output",
                        "--mode", "logpy_only",
                        "--seed", "0",
                        "--y_column_types", "categorical",
                        "--batch_size", str(args.batch_size),
                ])
                run_jolt(args=config_args, model=model, tokenizer=tokenizer)
                if shot == '0':
                    break  # for 0 shot, output is deterministic with logits, so only do 1 seed


if __name__ == '__main__':
    main()
