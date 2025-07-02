import torch
import os
import pickle
import time
import logging
import numpy as np
from tqdm import tqdm
from helpers import construct_prompts, process_generated_results
from hf_api import hf_generate
from helpers import get_predicted_values_from_generated_sample


def sample(args, tokenizer, model, results):   
    with torch.no_grad():
        # generate
        results['gen'] = [[] for _ in range(len(results['data']['x_test']))]
        # generate the prompts from the data
        prompts = construct_prompts(
            x_train=results['data']['x_train'],
            y_train=results['data']['y_train'],
            x_test=results['data']['x_test'],
            args=args,
            dim_y=results['dim_y']  
        )

        num_prompts = len(prompts)
        for idx in tqdm(range(num_prompts), desc='Sampling'):
            prompt = prompts[idx]
            samples = []
            num_samples = args.num_samples
            
            # Add timeout and retry limits to prevent infinite loops
            max_timeout_per_sample = 30.0  # 30 seconds max per sample
            max_failed_attempts = 3  # Max 3 failed attempts
            start_time = time.time()
            failed_attempts = 0
            
            while num_samples > 0:
                # Check timeout condition
                elapsed_time = time.time() - start_time
                if elapsed_time > max_timeout_per_sample:
                    logging.warning(f"JOLT sampling timeout ({max_timeout_per_sample}s) reached for sample {idx}. "
                                  f"Filling remaining {num_samples} samples with fallback value 0.0")
                    # Fill remaining samples with fallback value - match expected data types
                    for _ in range(num_samples):
                        fallback_sample = []
                        for col_idx in range(results['dim_y']):
                            if hasattr(args, 'y_column_types') and col_idx < len(args.y_column_types):
                                if args.y_column_types[col_idx] == 'categorical':
                                    # For categorical, use first available category as fallback
                                    if 'categories' in results and col_idx < len(results['categories']) and len(results['categories'][col_idx]) > 0:
                                        fallback_sample.append(results['categories'][col_idx][0])
                                    else:
                                        fallback_sample.append('unknown')
                                elif args.y_column_types[col_idx] == 'numerical':
                                    # For numerical, use appropriate numeric type
                                    if hasattr(args, 'num_decimal_places_y') and args.num_decimal_places_y == 0:
                                        fallback_sample.append(0)  # int
                                    else:
                                        fallback_sample.append(0.0)  # float
                                else:
                                    fallback_sample.append(0.0)
                            else:
                                fallback_sample.append(0.0)
                        samples.append(np.array(fallback_sample, dtype=object))
                    break
                
                # Check failed attempts limit
                if failed_attempts >= max_failed_attempts:
                    logging.warning(f"JOLT sampling max failed attempts ({max_failed_attempts}) reached for sample {idx}. "
                                  f"Filling remaining {num_samples} samples with fallback value 0.0")
                    # Fill remaining samples with fallback value - match expected data types
                    for _ in range(num_samples):
                        fallback_sample = []
                        for col_idx in range(results['dim_y']):
                            if hasattr(args, 'y_column_types') and col_idx < len(args.y_column_types):
                                if args.y_column_types[col_idx] == 'categorical':
                                    # For categorical, use first available category as fallback
                                    if 'categories' in results and col_idx < len(results['categories']) and len(results['categories'][col_idx]) > 0:
                                        fallback_sample.append(results['categories'][col_idx][0])
                                    else:
                                        fallback_sample.append('unknown')
                                elif args.y_column_types[col_idx] == 'numerical':
                                    # For numerical, use appropriate numeric type
                                    if hasattr(args, 'num_decimal_places_y') and args.num_decimal_places_y == 0:
                                        fallback_sample.append(0)  # int
                                    else:
                                        fallback_sample.append(0.0)  # float
                                else:
                                    fallback_sample.append(0.0)
                            else:
                                fallback_sample.append(0.0)
                        samples.append(np.array(fallback_sample, dtype=object))
                    break
                
                bs = min(args.batch_size, num_samples)
                res = hf_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_str=prompt,
                    batch_size=bs,
                    temp=args.temperature, 
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_generated_length
                )
                
                valid_samples_in_batch = 0
                for j in range(len(res)):
                    gen_sample = get_predicted_values_from_generated_sample(
                        generated_input=res[j],
                        args=args,
                        category_names=results['categories']
                        )
                    if gen_sample is not None:
                        samples.append(gen_sample)
                        num_samples -= 1
                        valid_samples_in_batch += 1
                
                # If no valid samples in this batch, increment failed attempts
                if valid_samples_in_batch == 0:
                    failed_attempts += 1
                else:
                    # Reset failed attempts on successful generation
                    failed_attempts = 0
                    
                del res
            results['gen'][idx] += samples

        # Print out the first sample.
        if args.print_prompts:
            for prompt, gen in zip(prompts, results['gen']):
                print(prompt, flush=True)
                print(f"> {gen[0]}", flush=True)
                print("\n==================================\n", flush=True)

    results['prompts'] = prompts

    results = process_generated_results(gen_results=results, args=args)

    # save off the results
    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results