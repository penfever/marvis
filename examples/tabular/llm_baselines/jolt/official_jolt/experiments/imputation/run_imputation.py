from parse_args import parse_command_line
from hf_api import get_model_and_tokenizer
from imputation_csv_parser import ImputationCsvParser
from run_jolt import run_jolt
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import tempfile
import contextlib


num_seeds = 3
missings = [0.1, 0.2, 0.3, 0.4]
use_country_column = False


def main():
    tmp_dir = './tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # parse the command line arguments
    args = parse_command_line()

    np.random.seed(args.seed)

    # get the llm and asociated tokenizer
    model, tokenizer = get_model_and_tokenizer(args)
    # load the dataset and get ground truth data
    csv_parser = ImputationCsvParser(
        csv_path=args.data_path,
        num_decimal_places=args.num_decimal_places_x,
        columns_to_ignore=args.columns_to_ignore,
        header_option=args.header_option
    )
    ground_truth_data = csv_parser.get_all_data()

    # create the mask
    # In the paris 2024 dataset, there are 91 rows and 3 columns to predict (Gold, Silver, Bronze)
    num_rows = ground_truth_data.shape[0]
    num_columns = ground_truth_data.shape[1] 
    # we don't want to mask the country column, so take 1 off the columns when creating the mask
    mask = np.random.choice([False, True], size=(num_rows, num_columns - 1), p=[1 - args.missing_fraction, args.missing_fraction])
    country_column = np.full(shape=(num_rows, 1), fill_value=False)
    mask = np.hstack((country_column, mask))

    column_names = csv_parser.get_column_names()
    ground_truth_values = []
    predictions = []
    logprobs = []
    joint_nlls = []
    for row in range(num_rows):
        print('Row = {}'.format(row))
        y_column_names = []
        for column, name in enumerate(column_names):
            if mask[row, column]:
                y_column_names.append(name)
                ground_truth_values.append(ground_truth_data[row, column])
        num_columns_to_impute = len(y_column_names)
        if num_columns_to_impute == 0:
            continue  # no missing values to impute in this row
        y_column_types = ['numerical'] * num_columns_to_impute  # all colums are numerical here

        data = csv_parser.get_data(
            y_column_names=y_column_names,
            missing_mask=mask,
            test_row=row
            )
        
        if data is not None:
            tmp_path = os.path.join('./tmp', next(tempfile._get_candidate_names()) + '.pkl')
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            continue

        column_names = csv_parser.get_rearranged_column_names()

        args.data_path = tmp_path
        args.y_column_names = y_column_names
        args.y_column_types = y_column_types
        if len(args.prefix) > 0:
            args.prefix = 'Each example contains four columns: {}, {}, {}, and {} that describe what type and how many medals a country won at the Paris 2024 olympics.'.format(column_names[0], column_names[1], column_names[2], column_names[3])

        results = run_jolt(args=args, model=model, tokenizer=tokenizer)

        for i in range(num_columns_to_impute):
            predictions.append(results['gen'][0][0][i])
            logprobs.append(results['metrics'][i]['y_logprobs'])
            joint_nlls.append(results['metrics'][0]['avg_nll'])

        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)

    # accumulate the metrics
    mae = mean_absolute_error(np.array(ground_truth_values), np.array(predictions))
    nll = -np.array(logprobs).mean()
    joint_nll = np.array(joint_nlls).mean()
    print("Final MAE = {}".format(mae))
    print("Final NLL = {}".format(nll))
    print("Final Joint NLL = {}".format(joint_nll))

    final_results = {
        'mae': mae,
        'nll': nll
    }

    with open(os.path.join(args.output_dir, args.experiment_name + '_final.pkl'), "wb") as g:
        pickle.dump(final_results, g)


if __name__ == '__main__':
    main()