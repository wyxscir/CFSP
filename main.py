import sys
import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_llama_flap import LlamaForCausalLM  # use flap need LlamaForCausalLM
from transformers import AutoTokenizer

from importlib.metadata import version

from lib.prune import prune_cfsp, prune_wanda_sp, prune_magnitude_sp, prune_flap_bias,check_sparsity
from lib.eval import eval_ppl


print('torch', version('torch'))  # 2.1.0
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(args, model, cache_dir="llm_weights"):
    if args.prune_method == "flap_bias":
        model = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            # trust_remote_code=True,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            # trust_remote_code=True,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    model.seqlen = 1024

    print(f" model.seqlen:{model.seqlen}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')

    parser.add_argument('--a', type=float, default=1, help='global control')
    parser.add_argument('--b', type=float, default=1, help='local control')
    parser.add_argument('--c', type=float, default=1, help='local control')
    parser.add_argument('--global_metrics', type=str, default="angular", help='angular, cosine, mse, mae, avg_base')
    parser.add_argument('--local_metrics', type=str, default="three_w_one_wa", help='one_wa, one_a, three_w_one_a, three_w_one_wa, wanda_base, mag_base')

    parser.add_argument('--cuda_friendly', action="store_true")

    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument("--prune_method", type=str, default="cfsp", choices=["wanda_sp", "mag_sp", "cfsp", "flap_bias"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()


    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if torch.cuda.is_available():
        print(" ---- CUDA is available! ------")
    else:
        print(" ---- no cuda! ------")


    # Prune the model
    print("pruning starts")
    if args.prune_method == "cfsp":
        print(f"loading llm model {args.model}")
        model = get_llm(args, args.model, args.cache_dir)
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        if "70B" in args.model or "70b" in args.model: # for 70b we use device_map to load onto multiple GPUs, thus the processing here.
            print(f"args.model: {args.model}")
            device = model.hf_device_map["lm_head"]
        print("use device ", device)

        prune_cfsp(args, model, tokenizer, device)

    elif args.prune_method == "flap_bias":
        print(f"loading llm model {args.model}")
        model = get_llm(args, args.model, args.cache_dir)
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        if "70B" in args.model or "70b" in args.model: # for 70b we use device_map to load onto multiple GPUs, thus the processing here.
            print(f"args.model: {args.model}")
            device = model.hf_device_map["lm_head"]
        print("use device ", device)

        prune_flap_bias(args, model, tokenizer, device)

    elif args.prune_method == "wanda_sp":
        print(f"loading llm model {args.model}")
        model = get_llm(args, args.model, args.cache_dir)
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        if "70B" in args.model or "70b" in args.model: # for 70b we use device_map to load onto multiple GPUs, thus the processing here.
            print(f"args.model: {args.model}")
            device = model.hf_device_map["lm_head"]
        print("use device ", device)
        prune_wanda_sp(args, model, tokenizer, device)

    else:
    # elif args.prune_method == "mag_sp":
        print(f"loading llm model {args.model}")
        model = get_llm(args, args.model, args.cache_dir)
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        if "70B" in args.model or "70b" in args.model: # for 70b we use device_map to load onto multiple GPUs, thus the processing here.
            print(f"args.model: {args.model}")
            device = model.hf_device_map["lm_head"]
        print("use device ", device)
        prune_magnitude_sp(args, model, tokenizer, device)


    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*"*30)


    if args.eval:
        ppl = eval_ppl(model, tokenizer, device)
        print(f"ppl on wikitext {ppl}")

    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)

        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()

