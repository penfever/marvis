import pickle
import numpy as np
from helpers import get_dimension, randomize, floats_to_str
from csv_parser import CsvParser


def fix_up_old_pickle_data(data):
    # To handle multiple y columns, we now expect all y columns to be in a list, even if
    #  there was only a single y column. If the y's are not in a list, put them in one.
    for key in ['y_train', 'y_test', 'y_true']:
        if key in data.keys():  # some files don't have y_true
            data[key] = np.expand_dims(data[key], axis=1)

    return data


def prepare_data(args):
    # load the data
    if ".pkl" in args.data_path:
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        if not args.impute_features:
            data = fix_up_old_pickle_data(data)
    elif ".csv" in args.data_path:  
        csv_parser = CsvParser(
            csv_path=args.data_path,
            y_column_names=args.y_column_names,
            y_column_types=args.y_column_types,
            header_option=args.header_option,
            num_decimal_places=args.num_decimal_places_x,
            shuffle=args.shuffle,
            seed=args.seed,
            columns_to_ignore=args.columns_to_ignore,
            missing_fraction=args.missing_fraction,
            column_separator=args.column_separator,
            name_value_separator=args.name_value_separator,
        )

        if args.csv_split_option == 'test_fraction':
            data = csv_parser.get_data_test_fraction(
                test_factor=args.test_fraction,
                train_size_limit=args.train_size_limit,
                test_size_limit=args.test_size_limit
            )
        elif args.csv_split_option == 'shots_per_class':
            data = csv_parser.get_data_shots_per_class(
                shots=args.shots,
                train_size_limit=args.train_size_limit,
                test_size_limit=args.test_size_limit,
            )
        elif args.csv_split_option == 'fixed_train_and_test_size':
            data = csv_parser.get_data_fixed_train_and_test_size(
                train_size_limit=args.train_size_limit,
                test_size_limit=args.test_size_limit
            )
        elif args.csv_split_option == 'fixed_indices':
            data = csv_parser.get_data_fixed_indices(
                train_start_index=args.train_start_index,
                train_end_index=args.train_end_index,
                test_start_index=args.test_start_index,
                test_end_index=args.test_end_index,
            )
        else:
            assert False
        if args.header_option == 'headers_as_prompt_prefix':
            # append the column header info to the prefix.
            column_header_text = "The data in each of the following examples is organized as follows: {}\n".format(
                csv_parser.get_column_names_concatenated())
            args.prefix += column_header_text
    else:
        assert False

    results = {'data': data, 'args': args}

    results['dim_x'] = get_dimension(results['data']['x_train'])
    if args.impute_features:
        results['dim_y'] = 0
    else:
        results['dim_y'] = len(args.y_column_types)

    # get the category names
    category_names = []
    key = 'y_train'
    if len(results['data']['y_train']) == 0:  # zero shot, no training data
        key = 'y_test'  # get the category names from the test set instead of the train set
    for i, column_type in enumerate(args.y_column_types):
        if column_type == 'categorical':
            if len(args.y_column_types) == 1:
                values = np.unique(results['data'][key]).tolist()
                values = [str(val) for val in values]
                category_names.append(values)
            else:
                values = np.unique(results['data'][key][:, i]).tolist()
                values = [str(val) for val in values]
                category_names.append(values)
        else:
            category_names.append([])
    results['categories'] = category_names

    return results