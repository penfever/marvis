from jsonargparse import ArgumentParser, ActionConfigFile
from hf_api import llm_map
from csv_parser import csv_header_options

def init_option_parser():
    parser = ArgumentParser()
    parser.add_argument('--cfg', action=ActionConfigFile, help='config file in YAML format')
    parser.add_argument("--mode", choices=["sample_logpy", "sample_only", "logpy_only"], default="sample_logpy",
                        help="Whether to sample or compute log likelihood, or both.")
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment.')
    parser.add_argument('--data_path', type=str, default=None, help='Path to .csv or .pkl file.')
    parser.add_argument('--llm_path', type=str, default=None, help='Path to LLM.')
    parser.add_argument("--llm_type", choices=llm_map.keys(), default="gemma-2-2B-instruct",
                        help="Hugging face model to use.")
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Path to directory where output results are written.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_decimal_places_x', type=int, default=0)
    parser.add_argument('--num_decimal_places_y', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--prefix', type=str, default='', help='Prompt prefix.')
    parser.add_argument('--break_str', type=str, default='\n', help='Break string between observed points.')
    parser.add_argument('--y_column_names', type=str, nargs='+', help='List of y column or target names.')
    parser.add_argument('--y_column_types', type=str, nargs='+', default=['numerical'],
        help='List of y column types, either "numerical" or "categorical".')
    parser.add_argument('--column_separator', type=str, default=';', help='Separator between different columns.')
    parser.add_argument('--name_value_separator', type=str, default=':', help='Separator between a column name and its value.')

    # sampling options
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to take at each test location.")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_generated_length', type=int, default=7)
    parser.add_argument('--top_k', type=int, default=None,
                        help="When generating a token, sample from only the top_k number of samples.")
    
    # debugging options
    parser.add_argument('--print_prompts', type=bool, default=False, help=('If true print out prompts.'))
    parser.add_argument('--print_sampling_rejections', type=bool, default=False, help=('If true print out details for when a sample is rejected.'))

    # csv input options
    parser.add_argument("--csv_split_option", choices=["test_fraction", "shots_per_class", "fixed_train_and_test_size", "fixed_indices"],
                        default="test_fraction", help="Various options for splitting a CSV file.")    
    parser.add_argument('--test_fraction', type=float, default=0.2, help='Fraction of the dataset to use as test examples.')
    parser.add_argument('--shots', type=int, default=None, help="Number of examples per class when y_is_categorical. If not None, overrides test fraction.")
    parser.add_argument('--header_option', type=str, default='no_headers', choices=csv_header_options,
                        help=('Options for using CSV column headers in the prompts.'))
    parser.add_argument('--train_size_limit', type=int, default=None, help="Maximum size of the CSV file training set.")
    parser.add_argument('--test_size_limit', type=int, default=None, help="Maximum size of the CSV file test set.")
    parser.add_argument('--train_start_index', type=int, default=None, help="Start index of the training set.")
    parser.add_argument('--train_end_index', type=int, default=None, help="End index (exclusive)of the training set.")
    parser.add_argument('--test_start_index', type=int, default=None, help="Start index of the test set.")
    parser.add_argument('--test_end_index', type=int, default=None, help="End index (exclusive) of the test set.")
    parser.add_argument('--missing_fraction', type=float, default=0.0, help="Fraction of missing data.")
    parser.add_argument('--impute_features', type=bool, default=False, help=('If True impute missing data.'))
    parser.add_argument('--shuffle', type=bool, default=True, help=('Whether or not to shuffle the data before splitting.'))
    parser.add_argument('--columns_to_ignore', nargs='+', default=[], help='List of columns in the CSV data to ignore.')

    return parser 


def parse_command_line():
    parser = init_option_parser()
    args = parser.parse_args()
    print(args, flush=True)
    return args