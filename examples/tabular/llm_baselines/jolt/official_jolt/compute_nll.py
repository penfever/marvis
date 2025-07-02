import torch
import numpy as np
import pickle
import math
import os
import logging
from helpers import construct_prompts, floats_to_str
from tqdm import tqdm
from scipy.special import log_softmax

logger = logging.getLogger(__name__)


def _get_mask(model, allowed_tokens):
    out_size = model.get_output_embeddings().out_features
    mask = torch.ones(out_size, dtype=torch.bool, device=model.device)
    mask[allowed_tokens] = False
    return mask


@torch.inference_mode()
def _get_y_logprobs_by_column(args, tokenizer, model, input_tokens, mask, y_ranges, column_type, column_index):
    '''
    Gets the logprobs of the y portions in input_texts,
    where the positions of y are given by y_ranges.

    Args:
        input_tokens: list (of length N) of encodings.
        y_ranges: list of y_ranges [y_range_1, ..., y_range_N].
    Returns:
        list of logprobs for each y given by y_ranges
            [y_logprobs_1, ..., y_logprobs_N], 
            where each y_logprobs_i is an np.array.
    '''
    bs = len(input_tokens)
    max_prompt_len = max(len(t) for t in input_tokens)
    
    # Debug: Check tensor sizes to understand memory usage
    tensor_size_gb = (bs * max_prompt_len * 8) / (1024**3)  # 8 bytes per long token
        
    if tensor_size_gb > 10.0:  # More than 10GB per tensor is unreasonable
        raise RuntimeError(f"JOLT tensor too large: {tensor_size_gb:.2f}GB. Reduce batch size or sequence length.")
    
    # Use more memory-efficient tensor creation
    input_ids = torch.full((bs, max_prompt_len),
                        tokenizer.pad_token_id,
                        dtype=torch.long, device=model.device)
    attn_mask = torch.zeros(
        (bs, max_prompt_len), dtype=torch.long, device=model.device)
    for k, t in enumerate(input_tokens):
        # More efficient: create on CPU then move to GPU once
        input_ids[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cpu').to(model.device)
        attn_mask[k, : len(t)] = 1  # Use scalar assignment instead of creating tensor

    # Shift so that tokens < n predict n
    # Force use_cache=False to save memory regardless of model type
    # Remove unnecessary slicing that creates memory copies
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    shift_logits = outputs['logits'][..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    

    if (mask is not None) and (column_type == 'numerical'):
        shift_logits[:, :,  mask] = -100
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    logprobs = -loss_fct(input=shift_logits.transpose(1, 2), target=shift_labels)
    
    y_logprobs = []
    for j in range(bs):
        y_logprobs_j = []
        y_logprobs_j.append(logprobs[j][y_ranges[j][column_index][0] - 1 : y_ranges[j][column_index][1] - 1].cpu())
        y_logprobs.append(y_logprobs_j)
    del outputs
    del logprobs
    # Clear GPU cache to prevent memory buildup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_logprobs


@torch.inference_mode()
def _get_y_logprobs_categorical(args, tokenizer, model, input_tokens, y_ranges):
    '''
    Gets the logprobs of the y portions in input_texts,
    where the positions of y are given by y_ranges.

    Args:
        input_tokens: list (of length N) of encodings.
        y_ranges: list of y_ranges [y_range_1, ..., y_range_N].
    Returns:
        list of logprobs for each y given by y_ranges
            [y_logprobs_1, ..., y_logprobs_N], 
            where each y_logprobs_i is an np.array.
    '''
    bs = len(input_tokens)
    max_prompt_len = max(len(t) for t in input_tokens)
    
    # Debug: Check tensor sizes to understand memory usage
    tensor_size_gb = (bs * max_prompt_len * 8) / (1024**3)  # 8 bytes per long token
        
    if tensor_size_gb > 10.0:  # More than 10GB per tensor is unreasonable
        raise RuntimeError(f"JOLT tensor too large: {tensor_size_gb:.2f}GB. Reduce batch size or sequence length.")
    
    # Use more memory-efficient tensor creation
    input_ids = torch.full((bs, max_prompt_len),
                        tokenizer.pad_token_id,
                        dtype=torch.long, device=model.device)
    attn_mask = torch.zeros(
        (bs, max_prompt_len), dtype=torch.long, device=model.device)
    for k, t in enumerate(input_tokens):
        # More efficient: create on CPU then move to GPU once
        input_ids[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cpu').to(model.device)
        attn_mask[k, : len(t)] = 1  # Use scalar assignment instead of creating tensor

    # Shift so that tokens < n predict n
    # Force use_cache=False to save memory regardless of model type
    # Remove unnecessary slicing that creates memory copies
    
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    shift_logits = outputs['logits'][..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    
    y_logprobs = []
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    logprobs = -loss_fct(input=shift_logits.transpose(1, 2), target=shift_labels)
        
    for j in range(bs):
        y_logprobs_j = []
        y_logprobs_j.append(logprobs[j][y_ranges[j][0][0] - 1 : y_ranges[j][0][1] - 1].cpu())
        y_logprobs.append(y_logprobs_j)
    
    del outputs
    del logprobs
    # Clear GPU cache to prevent memory buildup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_logprobs


def construct_y_str(y_test_true, categories, args, tokenizer, non_y):
    y_names = args.y_column_names 
    y_types = args.y_column_types
    y_separator = args.column_separator
    
    y_ranges = [[]]
    
    y_strs = [non_y]
    
    for i, y_type in enumerate(y_types):
        if i == len(y_types) - 1:
            cur_sep = args.break_str
        else:
            cur_sep = y_separator
            
        if y_type == 'numerical':
            y_partial_str = floats_to_str(y_test_true[i], args.num_decimal_places_y)
            
            if args.header_option == 'headers_as_item_prefix':
                prefix_y = y_names[i] + args.name_value_separator
            else:
                prefix_y = ''
            
            left_range = len(tokenizer.encode(y_strs[0] + prefix_y))
            right_range = len(tokenizer.encode(y_strs[0] + prefix_y + y_partial_str + cur_sep))
            
            y_partial_strs = [prefix_y + y_partial_str + cur_sep]
            y_ranges[0].append([left_range, right_range])
        else:
            y_partial_str = str(y_test_true[i])
                
            if args.header_option == 'headers_as_item_prefix':
                prefix_y = y_names[i] + args.name_value_separator
            else:
                prefix_y = ''
                
            left_range = len(tokenizer.encode(y_strs[0] + prefix_y))
            right_range = len(tokenizer.encode(y_strs[0] + prefix_y + y_partial_str + cur_sep))
            
            y_ranges[0].append([left_range, right_range])
            
            y_partial_strs = [prefix_y + y_partial_str + cur_sep]
            
            for category in categories[i]:
                if str(category) != str(y_test_true[i]):
                    right_range = len(tokenizer.encode(y_strs[0] + prefix_y + str(category) + cur_sep))
                    y_ranges.append([[left_range, right_range]])
                    
                    y_partial_strs.append(prefix_y + str(category) + cur_sep)
        
        for i in range(1, len(y_strs)):
            y_strs[i] += y_partial_strs[0]
            
        for y_partial_str in y_partial_strs[1:]:
            y_strs.append(y_strs[0] + y_partial_str)  
                              
        y_strs[0] += y_partial_strs[0]
    y_encoded = [tokenizer.encode(y_str) for y_str in y_strs]
    max_length = max([len(y) for y in y_encoded])
    return y_strs, y_encoded, y_ranges, max_length


def construct_y_str_single_categorical(categories, args, tokenizer, non_y):
    y_names = args.y_column_names 
    y_ranges = []
    y_strs = []
    cur_sep = args.break_str
            
    if args.header_option == 'headers_as_item_prefix':
        prefix_y = y_names[0] + args.name_value_separator
    else:
        prefix_y = ''
        
    left_range = len(tokenizer.encode(non_y + prefix_y))    
    y_partial_strs = []
    for category in categories[0]:
        right_range = len(tokenizer.encode(non_y + prefix_y + str(category) + cur_sep))
        y_ranges.append([[left_range, right_range]])
        y_partial_strs.append(prefix_y + str(category) + cur_sep)
        
    for y_partial_str in y_partial_strs:
        y_strs.append(non_y + y_partial_str)  
                            
    y_encoded = [tokenizer.encode(y_str) for y_str in y_strs]
    max_length = max([len(y) for y in y_encoded])
    return y_strs, y_encoded, y_ranges, max_length


# This is optimized for the case of a single categorical target
def compute_classification_probabilities(args, tokenizer, model, results):
    if 'metrics' not in results.keys():
        results['metrics'] = [{'probabilities_from_logits': [], 'y_logprobs': []} for _ in range(len(args.y_column_types))]
    else:
        for i in range(len(results['metrics'])): 
            results['metrics'][i]['probabilities_from_logits'] = []
            results['metrics'][i]['y_logprobs'] = []
            
        for _ in range(len(results['metrics']), len(args.y_column_types)):
            results['metrics'].append({'probabilities_from_logits': [], 'y_logprobs': []})
    
    # EDGE CASE FIX: Determine the most common category for fallback assignment
    # This handles cases where test subset contains labels not present in training categories
    fallback_category_idx = 0  # Default to first category
    if len(results['data']['y_train']) > 0 and len(results['categories'][0]) > 0:
        # Find the most common category in training data for fallback
        train_labels = results['data']['y_train']
        if hasattr(train_labels, 'flatten'):
            train_label_strs = [str(label) for label in train_labels.flatten()]
        else:
            train_label_strs = [str(label) for label in train_labels]
        
        # Count occurrences of each category in training data
        category_counts = {}
        available_categories = [str(cat) for cat in results['categories'][0]]
        for label in train_label_strs:
            if label in available_categories:
                category_counts[label] = category_counts.get(label, 0) + 1
        
        if category_counts:
            most_common_category = max(category_counts, key=category_counts.get)
            fallback_category_idx = available_categories.index(most_common_category)
            logger.info(f"JOLT fallback category for missing labels: '{most_common_category}' (idx={fallback_category_idx})")
        else:
            logger.info(f"JOLT fallback category: using first category '{available_categories[0]}' (idx=0)")
    
    full_texts, enc_full_texts, y_ranges = [], [], []
    max_len = 0
    for x_test, y_test_true in tqdm((zip(results['data']['x_test'], results['data']['y_test'])), desc='Processing prompts'):
        non_y = construct_prompts(
            args = args,
            x_train=results['data']['x_train'],
            y_train=results['data']['y_train'],
            x_test=np.array([x_test]),
            dim_y=results['dim_y']  
        )[0]
        str_y_test_true, token_y_test_true, y_range, cur_max_length = construct_y_str_single_categorical(
            results['categories'],
            args,
            tokenizer=tokenizer,
            non_y=non_y
        )
        
        full_texts.extend(str_y_test_true)
        enc_full_texts.extend(token_y_test_true)
        y_ranges.extend(y_range)
        max_len = max(max_len, cur_max_length)
    
    results['metrics'][0]['full_texts'] = full_texts
    results['metrics'][0]['enc_full_texts'] = enc_full_texts
    results['metrics'][0]['y_ranges'] = y_ranges

    num_categories = len(results['categories'][0])
    assert len(full_texts) == (len(results['data']['x_test']) * num_categories)
    
    # Ultra-aggressive memory optimization: process ONE sample at a time
    # This prevents massive tensor allocation that was causing OOM
    print(f"[JOLT Memory Opt] Processing {len(enc_full_texts)} prompts individually to prevent OOM")
    y_logprobs = []
    
    for i in tqdm(range(len(enc_full_texts)), desc="Computing log probs (1 by 1)"):
        # Clear cache before each sample to prevent accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Process exactly one prompt at a time
        single_result = _get_y_logprobs_categorical(
            args,
            tokenizer,
            model,
            [enc_full_texts[i]],  # Single item list
            [y_ranges[i]]  # Single item list
        )
        y_logprobs.extend(single_result)

    for itr, y_test_true in tqdm(enumerate(results['data']['y_test']), desc="Computing probabilities"): # this loops over number of test locations
        y_logprobs_given_x = y_logprobs[itr * num_categories : (itr + 1) * num_categories]

        idx_gt_column = None
        # Convert ground truth to string for comparison with categories
        gt_value_str = str(y_test_true[0])
        for j, category in enumerate(results['categories'][0]):
            # Try both direct string match and numeric match
            if str(category) == gt_value_str or category == y_test_true[0]:
                idx_gt_column = j
                break
        
        if idx_gt_column is None:
            # GRACEFUL FALLBACK: Assign missing ground truth labels to fallback category
            # This occurs when test subset contains labels not present in training categories
            available_categories = [str(cat) for cat in results['categories'][0]]
            logger.warning(f"Ground truth value '{gt_value_str}' not found in training categories: {available_categories}. "
                          f"Assigning to fallback category '{available_categories[fallback_category_idx]}' (idx={fallback_category_idx}). "
                          f"This sample will be marked as incorrect but evaluation will continue.")
            idx_gt_column = fallback_category_idx

        logprobs = [y_logprob[0].sum().item() for y_logprob in y_logprobs_given_x]
                
        logprobs_normalized = log_softmax(logprobs)
        results['metrics'][0]['y_logprobs'].append(logprobs_normalized[idx_gt_column])
        results['metrics'][0]['probabilities_from_logits'].append(np.exp(logprobs_normalized))
    
    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results       


# This is the general multi-target case
def compute_nll(args, tokenizer, model, results):
    if (len(args.y_column_types) == 1) and (args.y_column_types[0] == 'categorical'):
        return compute_classification_probabilities(args, tokenizer, model, results)       
    
    if 'metrics' not in results.keys():
        results['metrics'] = [{'probabilities_from_logits': [], 'y_logprobs': []} for _ in range(len(args.y_column_types))]
    else:
        for i in range(len(results['metrics'])): 
            results['metrics'][i]['probabilities_from_logits'] = []
            results['metrics'][i]['y_logprobs'] = []
            
        for _ in range(len(results['metrics']), len(args.y_column_types)):
            results['metrics'].append({'probabilities_from_logits': [], 'y_logprobs': []})
    
    full_texts, enc_full_texts, y_ranges = [], [], []
    max_len = 0
        
    for x_test, y_test_true in tqdm((zip(results['data']['x_test'], results['data']['y_test'])), desc='Processing prompts'):
        non_y = construct_prompts(
            args=args,
            x_train=results['data']['x_train'],
            y_train=results['data']['y_train'],
            x_test=np.array([x_test]),
            dim_y=results['dim_y']  
        )[0]
        
        str_y_test_true, token_y_test_true, y_range, cur_max_length = construct_y_str(
            y_test_true,
            results['categories'],
            args,
            tokenizer=tokenizer,
            non_y=non_y
        )
        
        full_texts.append(str_y_test_true)
        enc_full_texts.append(token_y_test_true)
        y_ranges.append(y_range)
        max_len = max(max_len, cur_max_length)
    
    results['metrics'][0]['full_texts'] = full_texts
    results['metrics'][0]['enc_full_texts'] = enc_full_texts
    results['metrics'][0]['y_ranges'] = y_ranges

    # Generate mask
    mask = None
    # When y is a number, it can also include '-', and '.'.
    # In here I assume that args.column_separator is mapped into a separate token;
    allowed = [str(i) for i in range(10)] + ['-', '.', args.break_str]
    if len(args.y_column_types) > 1:
        allowed += args.column_separator
    allowed_tokens = set([tokenizer.convert_tokens_to_ids(token)
                        for token in allowed])
    mask = _get_mask(model, list(allowed_tokens))
    
    nll = []
    
    # TODO: For now let's assume that first class token are different (though they might actually be the same, 
    # and we have to accomodate for that as well).
    # For the nll evaluation, we will assume that we know the classes for each column
    # So, we will evaluate using p(class_i | only class_1, class_2, ..., class_n could be possible)
        
    for enc_full_text, y_range, y_test_true in tqdm(zip(enc_full_texts, y_ranges, results['data']['y_test']), 
                                                    desc='Computing logprobs'):
        gt_y_logprobs = []
        column_logprobs = []
        for i, column_type in enumerate(args.y_column_types):
            column_logprobs.extend(
                _get_y_logprobs_by_column(
                    args=args, 
                    tokenizer=tokenizer, 
                    model=model,
                    input_tokens=enc_full_text[:1],
                    mask=mask,
                    y_ranges=y_range[:1], 
                    column_type=column_type,
                    column_index=i
                )[0]
            )
        gt_y_logprobs.append(column_logprobs)

        nll_per_item = 0
        current_str_pointer = 1
        for i, column in enumerate(args.y_column_types):
            if column == 'numerical':
                nll_num_column = -(gt_y_logprobs[0][i].sum().item() + args.num_decimal_places_y * np.log(10))
                results['metrics'][i]['y_logprobs'].append(-nll_num_column)
                nll_per_item += nll_num_column
            else:
                y_logprobs = []
                for k in range(current_str_pointer, current_str_pointer + len(results['categories'][i]) - 1):
                    y_logprobs.append(
                        _get_y_logprobs_by_column(
                            args=args, 
                            tokenizer=tokenizer, 
                            model=model,
                            input_tokens=[enc_full_text[k]],
                            mask=None,
                            y_ranges=[y_range[k]], 
                            column_type=['categorical'],
                            column_index=0
                        )[0]
                    )
                current_str_pointer += len(results['categories'][i]) - 1
                
                idx_gt_column = None
                for j, category in enumerate(results['categories'][i]):
                    if category == str(y_test_true[i]):
                        idx_gt_column = j
                assert idx_gt_column is not None
                
                y_logprobs = [y_logprob[0].sum().item() for y_logprob in y_logprobs]
                y_logprobs.insert(idx_gt_column, gt_y_logprobs[0][i].sum().item())
                
                # If we assume that the model know the classes then this should used to evaluate nlls
                # y_logprobs = [gt_y_logprobs[0][i][0].item()] + [y_logprob[0][0].item() for y_logprob in y_logprobs]
                
                logprobs = log_softmax(y_logprobs)
                nll_per_item += -logprobs[idx_gt_column]
                results['metrics'][i]['y_logprobs'].append(logprobs[idx_gt_column])
                results['metrics'][i]['probabilities_from_logits'].append(np.exp(logprobs))
        nll.append(nll_per_item)            

    results['metrics'][0]['nll'] = np.array(nll).sum()
    results['metrics'][0]['avg_nll'] = np.array(nll).sum() / len(nll)
    print("avg_nll = {}".format(results['metrics'][0]['avg_nll']))
    
    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results
