import torch
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

llm_map = {
    "llama-2-7B": "meta-llama/Llama-2-7b",
    "llama-2-70B": "meta-llama/Llama-2-70b",
    "llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "llama-3-70B": "meta-llama/Meta-Llama-3-70B",
    "llama-3.1-70B-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7B-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
    "phi-3-small-128k-instruct": "microsoft/Phi-3-small-128k-instruct",
    "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "phi-3.5-moe-instruct": "microsoft/Phi-3.5-MoE-instruct",
    "gemma-2-27B": "google/gemma-2-27b",
    "gemma-2-27B-instruct": "google/gemma-2-27b-it",
    "gemma-2-9B": "google/gemma-2-9b",
    "gemma-2-9B-instruct": "google/gemma-2-9b-it",
    "gemma-2-2B": "google/gemma-2-2b",
    "gemma-2-2B-instruct": "google/gemma-2-2b-it",
    "qwen2.5-1.5B-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7B-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-72B-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-r1-distill-qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
}


DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(llm_path, llm_type):
    if llm_path is None:
        llm_path = llm_map[llm_type]
    if "llama-2" in llm_type:
        tokenizer = LlamaTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            padding_side="left"
        )
    elif "llama-3" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path
        )
    elif "phi-3" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
        )
    elif "mixtral" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )
    elif "gemma" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )
    elif "qwen" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )
    else:
        assert False

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if llm_type != "phi-3-small-128k-instruct":
        tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model_and_tokenizer(args):
    if args.llm_path is None:
        llm_path = llm_map[args.llm_type]
    else:
        llm_path = args.llm_path
    tokenizer = get_tokenizer(llm_path, args.llm_type)
    
    # Determine device and dtype based on availability
    use_cuda = torch.cuda.is_available() and getattr(args, 'device', 'auto') != 'cpu'
    device_map = "auto" if use_cuda else "cpu"
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32
    
    # Override dtype for specific models if needed
    if "llama-2" in args.llm_type:
        torch_dtype = torch.float16 if use_cuda else torch.float32
        model = LlamaForCausalLM.from_pretrained(
            llm_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    elif "llama-3" in args.llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    elif "phi-3" in args.llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map=device_map,
            torch_dtype="auto" if use_cuda else torch.float32,
            trust_remote_code=True,
        )
    elif "mixtral" in args.llm_type:
        torch_dtype = torch.float16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    elif "gemma" in args.llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    elif "qwen" in args.llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        assert False

    model.eval()
    return model, tokenizer


def _generate_core(
    model,
    tokenizer,
    batch,
    temp,
    top_p,
    top_k,
    max_new_tokens,
    num_input_ids
    ):
    args = {
        'do_sample': True,
        'max_new_tokens': max_new_tokens,
        'temperature': temp,
        'renormalize_logits': False,
        'pad_token_id': tokenizer.eos_token_id
    }
    if top_k is not None:
        args['top_k'] = top_k
    else:
        args['top_p'] = top_p
        args['top_k'] = 0

    generate_ids = model.generate(
        **batch,
        **args
    )

    gen_strs = tokenizer.batch_decode(
        generate_ids[:, num_input_ids:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return gen_strs


# This assumes that there is only a single prompt and it gets replicated batch_size times
@torch.inference_mode()
def hf_generate(
    model,
    tokenizer,
    input_str,
    batch_size,
    temp, 
    top_p,
    top_k,
    max_new_tokens
    ):
    batch = tokenizer([input_str], return_tensors="pt")
    batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
    # Use the device of the model instead of hardcoding CUDA
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    num_input_ids = batch['input_ids'].shape[1]

    return _generate_core(model, tokenizer, batch, temp, top_p, top_k, max_new_tokens, num_input_ids)
