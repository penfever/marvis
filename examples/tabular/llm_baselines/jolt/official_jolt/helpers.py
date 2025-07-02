import re
import decimal
import numpy as np
import sys
from sklearn.metrics import accuracy_score, roc_auc_score

ctx = decimal.Context()
ctx.prec = 20


def y_is_single_categorical(y_column_types):
    # return True only if this is a simple classification scenario (no multi-label, no regression)
    return ('categorical' in y_column_types) and (len(y_column_types) == 1)


def y_is_pure_regression(y_column_types):
    return False if 'categorical' in y_column_types else True


def get_predicted_values_from_generated_sample(generated_input, args, category_names):
    # sanity checks
    if args.break_str not in generated_input:
        if args.print_sampling_rejections:
            print("No break string in the generated sample.")
            print(generated_input)
        return None  # throw away sample if it doesn't contain a break_str

    has_column_names = False
    expected_num_y_columns = len(args.y_column_types)
    num_y_column_names = 0 if (
        (args.y_column_names == None) or
        (args.header_option == 'no_headers') or
        (args.header_option == 'headers_as_prompt_prefix')) else len(args.y_column_names)
    assert expected_num_y_columns > 0
    if num_y_column_names > 0:
        has_column_names = True
        assert expected_num_y_columns == num_y_column_names
        
    generated_till_break = re.split(args.break_str, generated_input)[0]
    # split the generated response into separate columns
    column_values = re.split(args.column_separator, generated_till_break)
    # the last value should be terminated by the break string, remove it
    if len(column_values) < expected_num_y_columns:
        if args.print_sampling_rejections:
            print("Number of column values={} in the generated response is fewer than expected={}.".format(column_values, expected_num_y_columns))
            print(generated_input)
        return None
    if len(column_values) > expected_num_y_columns:
        column_values = column_values[-expected_num_y_columns :]  # take the last expected_num_y_columns

    # remove the class names if they are there
    if has_column_names:
        for i in range(expected_num_y_columns):
            column_values[i] = re.split(args.name_value_separator, column_values[i])[-1]

    # check that the extracted values conform with their type (numerical or categorical)
    results = []
    for i in range(expected_num_y_columns):
        if args.y_column_types[i] == 'categorical':
            assert category_names is not None
            if column_values[i] in category_names[i]:
                results.append(column_values[i])
            else:
                if args.print_sampling_rejections:
                    print("Generated response for column={} does not contain a known category.".format(i))
                    print(generated_input)
                return None
        elif args.y_column_types[i] == 'numerical':
            try:
                number = re.findall(r'-?\d+\.?\d*', column_values[i])[0]
            except:
                if args.print_sampling_rejections:
                    print("Exception: Generated response for column={} does not contain a number.".format(i))
                    print(generated_input)
                return None
            
            if not number:
                if args.print_sampling_rejections:
                    print("Generated response for column={} does not contain a number.".format(i))
                    print(generated_input)
                return None  # if the generataion does not contain a number, return None, throw away sample
            else:
                if args.num_decimal_places_y == 0:
                    results.append(int(number))
                else:
                    results.append(float(number))
        else:
            sys.exit()

    return np.array(results, dtype=object)
      

def _map_to_ordinal(array, ordering):
    if ordering is not None:
        return np.array([ordering[key] for key in array])
    else:
        return array


def randomize(x, y):
    permutation = np.random.permutation(len(x))
    return  (np.array(x))[permutation], (np.array(y))[permutation]


def sequential_sort(x, y, x_ordering):
    sort_indices = np.argsort(np.array(_map_to_ordinal(x, x_ordering)))
    return (np.array(x))[sort_indices], (np.array(y))[sort_indices]


def get_dimension(a):
    if a.ndim > 1:
        return a.shape[1] # return the second dimension size
    else:
        return 1

def _float_to_str(f, num_decimal=None):
    if isinstance(f, np.int64):
        f = int(f)
    """Convert float to string without resorting to scientific notation."""
    if isinstance(f, np.floating):
        f = float(f)
    
    d1 = ctx.create_decimal(repr(f))
    if num_decimal is not None:
        d1 = round(d1, num_decimal)
    return format(d1, 'f')


def floats_to_str(nums, num_decimal, dim=1):
    if np.ndim(nums) == 0:
        return _float_to_str(nums, num_decimal)  # when y_dim = 1, only a scalar is passed
    if dim > 1:  # can have multiple dimensions in x and y
        return [[_float_to_str(value, num_decimal) for value in group] for group in nums]
    else:
        return [_float_to_str(num, num_decimal) for num in nums]


def _format_observed_data_point(
    x, 
    y, 
    dim_y,  
    break_str,
    column_separator='; ', 
    y_names=None, 
    name_value_separator=':'
):
    y_point_string = ''
    for i in range(dim_y):
        if y_names is not None:
            y_point_string += y_names[i] + name_value_separator
        y_point_string += str(y[i])
        if i < dim_y - 1:
            y_point_string += column_separator
    return f'{x}{y_point_string}{break_str}'
    

def _format_query_data_point(x):
    return f'{x}'


def construct_prompts(
        x_train,
        y_train,
        x_test,
        args,
        dim_y
):
    # Convert xy train and x test to str
    str_x_train = x_train
    str_x_test = x_test

    str_y_train = np.empty(y_train.shape, dtype=object)
    for i, column_type in enumerate(args.y_column_types):
        if column_type == 'numerical':
            str_y_train[:, i] = np.expand_dims(np.array(floats_to_str(nums=y_train[:, i], num_decimal=args.num_decimal_places_y, dim=1)), axis=0)         
        else:  # categorical, already a string
            str_y_train[:, i] = y_train[:, i]

    # note:
    # we assume that the input training data is already in random order,
    # so we just need to construct the base prompt here
    base_prompt = args.prefix
    for x, y in zip(str_x_train, str_y_train):
        base_prompt += _format_observed_data_point(
            x=x,
            y=y,
            dim_y=dim_y,
            break_str=args.break_str, 
            column_separator=args.column_separator,
            y_names=args.y_column_names if args.header_option == 'headers_as_item_prefix' else None,
        )

    prompts = []
    for xt_str in str_x_test:
        prompt = f'{base_prompt}{_format_query_data_point(x=xt_str)}'
        prompts.append(prompt)
    return prompts


def process_generated_results(gen_results, args):
    per_column_metrics = []
    for column, y_column_type in enumerate(args.y_column_types):
        metrics = {}
        if y_column_type == 'numerical':
            # Get all sampled y values. Shape is (num ys, num samples).
            num_xs = len(gen_results['data']['x_test'])
            y_tests = [[] for _ in range(num_xs)]
            y_test_mean = [np.nan for _ in range(num_xs)]
            y_test_median = [np.nan for _ in range(num_xs)]
            y_test_std = [np.nan for _ in range(num_xs)]
            y_test_lower = [np.nan for _ in range(num_xs)]
            y_test_upper = [np.nan for _ in range(num_xs)]
            for i in range(len(gen_results['gen'])):
                ys = gen_results['gen'][i]
                y_tests[i] += ys
                if len(args.y_column_types) > 1:
                    ys = np.array(ys, dtype=object)
                    y_test_mean[i] = np.mean(ys[:, column])
                    y_test_median[i] = np.median(ys[:, column])
                    y_test_std[i] = np.std(ys[:, column])
                    y_test_lower[i] = np.percentile(ys[:, column], 2.5)
                    y_test_upper[i] = np.percentile(ys[:, column], 97.5)
                else:
                    y_test_mean[i] = np.mean(ys)
                    y_test_median[i] = np.median(ys)
                    y_test_std[i] = np.std(ys)
                    y_test_lower[i] = np.percentile(ys, 2.5)
                    y_test_upper[i] = np.percentile(ys, 97.5)

            if len(args.y_column_types) > 1:
                mae = np.mean(np.abs(y_test_median - np.array(gen_results['data']['y_test'])[:, column]))
            else:
                mae = np.mean(np.abs(y_test_median - np.array(gen_results['data']['y_test'])))

            metrics['y_test'] = y_tests
            if (len(args.y_column_types) == 1) and (args.y_column_types[0] == 'numerical'):  # only used in black box opt with one output y
                metrics['y_test_max_x'] = gen_results['data']['x_test'][np.argmax(np.max(np.array(y_tests), axis=1))]  # find argmax of the largest sample
            metrics['y_test_mean'] = y_test_mean
            metrics['y_test_median'] = y_test_median
            metrics['y_test_std'] = y_test_std
            metrics['y_test_lower'] = y_test_lower
            metrics['y_test_upper'] = y_test_upper
            metrics['mae'] = mae
            per_column_metrics.append(metrics)
            print(f'mae: {mae}')
        else:  # categorical
            num_xs = len(gen_results['data']['x_test'])
            y_tests = [[] for _ in range(num_xs)]
            y_test_distribution = [[] for _ in range(num_xs)]
            y_test_mode = [np.nan for _ in range(num_xs)]
            probabilities_from_sampling = [np.nan for _ in range(num_xs)]
            for i in range(len(gen_results['gen'])):
                # count the occurances of each class
                y_tests[i] += gen_results['gen'][i]
                if len(args.y_column_types) > 1:
                    ys = np.array(gen_results['gen'][i], dtype=object)
                    for category in gen_results['categories'][column]:
                        y_test_distribution[i].append((np.array(ys[:, column]) == category).sum())
                    y_test_mode[i] = gen_results['categories'][column][np.argmax(y_test_distribution[i])]
                    probabilities_from_sampling[i] = y_test_distribution[i] / np.array(y_test_distribution[i]).sum()
                else:
                    for category in gen_results['categories'][column]:
                        y_test_distribution[i].append((np.array(gen_results['gen'][i]) == category).sum())
                    y_test_mode[i] = gen_results['categories'][column][np.argmax(y_test_distribution[i])]
                    probabilities_from_sampling[i] = y_test_distribution[i] / np.array(y_test_distribution[i]).sum()

            # compute accuracy
            if len(args.y_column_types) > 1:
                accuracy_sampling = accuracy_score(np.array(gen_results['data']['y_test'], dtype=str)[:, column], y_test_mode)
            else:
                accuracy_sampling = accuracy_score(np.array(gen_results['data']['y_test']), y_test_mode)
                if len(gen_results['categories'][column]) == 2:  # binary classification
                    auc = roc_auc_score(np.array(gen_results['data']['y_test']), np.array(probabilities_from_sampling)[:, 1])
                    metrics['auc_sampling'] = auc
                    print(f'sampling auc: {auc}')
            metrics['y_test_mode'] = y_test_mode
            metrics['y_test_distribution'] = y_test_distribution
            metrics['probabilities_from_sampling'] = probabilities_from_sampling
            metrics['accuracy_sampling'] = accuracy_sampling
            metrics['y_test'] = y_tests
            per_column_metrics.append(metrics)
            print(f'sampling accuracy: {accuracy_sampling}')

    gen_results['metrics'] = per_column_metrics
    return gen_results
    

def map_text_labels_to_integer_labels(text_labels, categories):
    integers = []
    for text_label in text_labels:
        integers.append(categories.index(text_label))
    integers = np.array(integers)
    return integers


def compute_classification_metrics(results, column_index):
    # Handle both 1D and 2D probability arrays
    probs = np.array(results['metrics'][column_index]['probabilities_from_logits'])
    
    if probs.ndim == 1:
        # If 1D, we have a single sample - expand to 2D
        probs = probs.reshape(1, -1)
    elif probs.ndim == 2:
        # If already 2D, check if we need to transpose
        if probs.shape[1] == 1 and probs.shape[0] > 1:
            # If shape is [n_classes, 1], transpose to [1, n_classes]
            probs = probs.T
    
    # Get predicted classes
    if probs.shape[0] == 1:
        # Single prediction case
        pred_indices = [np.argmax(probs[0])]
    else:
        # Multiple predictions case
        pred_indices = np.argmax(probs, axis=1)
    
    results['metrics'][column_index]['y_test_mode'] = np.array(results['categories'][column_index])[pred_indices]
    
    # accuracy
    accuracy_logits = accuracy_score(
        np.array(results['data']['y_test'][:, column_index], dtype=results['metrics'][column_index]['y_test_mode'].dtype),
                 results['metrics'][column_index]['y_test_mode']
    )
    results['metrics'][column_index]['accuracy_logits'] = accuracy_logits
    print(f'logits accuracy for column {column_index}: {accuracy_logits}')

    # AUC
    if len(results['categories'][column_index]) > 2:  # multi-class
        try:
            auc = roc_auc_score(
                np.array(results['data']['y_test'][:, column_index], dtype=results['metrics'][column_index]['y_test_mode'].dtype),
                np.array(results['metrics'][column_index]['probabilities_from_logits']),
                multi_class='ovr',
                average='macro'
            )
        except Exception:
            auc = np.nan
            print("Error computing AUC. Setting to NaN.")
    else:  # binary
        auc = roc_auc_score(
            np.array(results['data']['y_test'][:, column_index], dtype=results['metrics'][column_index]['y_test_mode'].dtype),
            np.array(results['metrics'][column_index]['probabilities_from_logits'])[:, 1]
        )
    results['metrics'][column_index]['auc_logits'] = auc
    print(f'logits auc for {column_index}: {auc}')

    return results
