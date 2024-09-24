import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT, BiasGPT

from .data import get_loaders
import math
from tqdm import tqdm
import sys
from .block_metrics import block_influence
import numpy as np



def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    check_sparsity for llama3

    Args:
        model (nn.Module): The model to check.

    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    # print(intermediate_size)
    # print(hidden_size)

    count = 0.0
    total_params = 0.0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0

        for name in subset:
            # print(name)
            W = subset[name].weight.data
            sub_count += W.numel()
            # print(W.numel())

            count += W.numel()

            if name == 'self_attn.q_proj' or name == 'self_attn.o_proj':

                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size

            elif name == 'self_attn.k_proj' or name == 'self_attn.v_proj':

                total_params += (hidden_size * hidden_size / 8)
                sub_params += (hidden_size * hidden_size / 8)

            else:

                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")


    model.config.use_cache = use_cache
    return float(count)/total_params



def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)  # 2048 is the upper limit.

    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def compress(layer, mlp_mask, device):

    mlp_mask = mlp_mask.to(device)

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    print(layer.mlp.intermediate_size)

    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()



# for flap
def compress_bias(layer, mlp_mask, mlp_mean_inp, device):

    bias = True

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    if bias:
        # Add the additional bias to compensate for the loss
        output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    if bias:
        # Re-initialize the Linear layer with new shape and bias
        layer.mlp.down_proj.in_features = mlp_mask.sum().item()
        # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
        layer.mlp.down_proj.bias.data = output_bias

    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()




def prune_cfsp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    our method
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers


    mlp_metric_list = []
    mlp_mask = []

    layer_importances = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer_importance = 0.0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

                if args.global_metrics == 'angular':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='angular').sum().cpu().item()
                elif args.global_metrics == 'cosine':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='cosine').sum().cpu().item()
                elif args.global_metrics == 'mse':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='mse').sum().cpu().item()
                elif args.global_metrics == 'mae':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='mae').sum().cpu().item()
                else:
                    layer_importance += 100



        layer_importances.append(layer_importance)
        for h in handles:
            h.remove()

        for name in subset:
            if args.local_metrics == "wanda_base":
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            elif args.local_metrics == "mag_base":
                W_metric = torch.norm(subset[name].weight.data, dim=0)

            elif args.local_metrics == "one_a":
                W = subset[name].weight.data
                W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**0.5

            elif args.local_metrics == "three_w_one_a":
                W = subset[name].weight.data
                # W_down: torch.Size([4096, 14336])
                # W_up: torch.Size([14336, 4096])
                # W_gate: torch.Size([14336, 4096])
                W_up =  find_layers(layer)['mlp.up_proj'].weight.data
                W_gate = find_layers(layer)['mlp.gate_proj'].weight.data
                W_up = W_up.t()
                W_gate = W_gate.t()
                W_metric = ((torch.abs(W)/torch.sum(torch.abs(W), dim=0)) +
                            (torch.abs(W_up)/torch.sum(torch.abs(W_up), dim=0))+
                            (torch.abs(W_gate)/torch.sum(torch.abs(W_gate), dim=0))) \
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            elif args.local_metrics == "three_w_one_wa":
                W = subset[name].weight.data
                # W_down: torch.Size([4096, 14336])
                # W_up: torch.Size([14336, 4096])
                # W_gate: torch.Size([14336, 4096])
                W_under = (torch.abs(W) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.b)
                W_up =  find_layers(layer)['mlp.up_proj'].weight.data
                W_gate = find_layers(layer)['mlp.gate_proj'].weight.data
                W_up = W_up.t()
                W_gate = W_gate.t()
                W_metric = ((torch.abs(W_under)/torch.sum(torch.abs(W_under), dim=0)) +
                            (torch.abs(W_up)/torch.sum(torch.abs(W_up), dim=0))+
                            (torch.abs(W_gate)/torch.sum(torch.abs(W_gate), dim=0))) \
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            elif args.local_metrics == "one_wa":
                W = subset[name].weight.data
                W_under = (torch.abs(W) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.b)
                W_metric = (torch.abs(W_under)/torch.sum(torch.abs(W_under), dim=0)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c


            W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())

            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer

        torch.cuda.empty_cache()


    layer_importances_sorted = sorted(enumerate(layer_importances), key=lambda x: x[1], reverse=True)

    for i in range(len(layer_importances_sorted)):
        index2 = layer_importances_sorted[i][0]
        number2 = layer_importances_sorted[i][1]
        print(f"layer: {index2} , importance: {number2} ")

    print(f"{args.global_metrics} layer_importances_sorted: {layer_importances_sorted}")


    def sigmoid(x):
        return 1 / (1 + np.exp(-x*args.a))

    layer_importances_mid = sum(layer_importances) / len(layer_importances)

    layer_importances = [(i-layer_importances_mid)/1e4 for i in layer_importances]
    layer_importances = [sigmoid(i) for i in layer_importances]


    avg = sum(layer_importances) / len(layer_importances)
    max_score = max(layer_importances)
    if max_score / avg * (1-args.pruning_ratio) >= 1:
        #
        scale_factor = (avg * (1 / (1-args.pruning_ratio) - 1)) /  (max_score - avg) / 1.05
        for i in range(len(layer_importances)):
            if layer_importances[i] > avg:
                layer_importances[i] = avg + (layer_importances[i] - avg) * scale_factor
            else:
                layer_importances[i] = avg - (avg - layer_importances[i]) * scale_factor
        avg = sum(layer_importances) / len(layer_importances)


    mlp_metric = torch.stack(mlp_metric_list)

    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)
    print(sorted_mlp_metric.shape)

    every_pruning_ratios = [i/avg*(1-args.pruning_ratio) for i in layer_importances]
    print(f"every_pruning_ratios: {every_pruning_ratios}")



    if args.cuda_friendly:
        thresholds = torch.tensor([
            sorted_mlp_metric[i][int(((sorted_mlp_metric.shape[1]*every_pruning_ratios[i])+64)/128)*128-1]
                                   for i in range(len(every_pruning_ratios))
                                   ])
        print(f"thresholds: {thresholds}")

    else:
        thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*every_pruning_ratios[i])] for i in range(len(every_pruning_ratios))])
        print(f"thresholds: {thresholds}")


    mlp_mask = (mlp_metric.t() >= thresholds).t()
    print(mlp_mask.shape)

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()





def prune_flap_bias(args, model, tokenizer, device=torch.device("cuda:0")):

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
    }

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    layers = model.model.layers

    mlp_metric_list = []
    mlp_baseline_inp_list = []
    mlp_mask = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):

        layer = layers[i]

        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], "WIFV")

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:

            W_metric = metrics['WIFV'](wrapped_layers, subset, name)
            mlp_metric_list.append(W_metric.cpu())
            mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        inps, outs = outs, inps

        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    mlp_metric = torch.stack(mlp_metric_list)
    mlp_metric = standarlization(mlp_metric)

    sorted_prune, indices = torch.sort(mlp_metric.view(-1), descending=True)
    threshold = sorted_prune[int(sorted_prune.shape[0]*(1 - args.pruning_ratio))]
    mlp_mask = (mlp_metric > threshold)


    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress_bias(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress_bias(model.model.layers[idx], mlp_mask[idx], mlp_baseline_inp_list[idx], device)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()






def prune_wanda_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=128,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers


    mlp_metric_list = []
    mlp_mask = []


    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer

        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
    mlp_metric = torch.stack(mlp_metric_list)
    mlp_metric = standarlization(mlp_metric)
    sorted_prune, indices = torch.sort(mlp_metric.view(-1), descending=True)
    threshold = sorted_prune[int(sorted_prune.shape[0]*(1 - args.pruning_ratio))]
    mlp_mask = (mlp_metric > threshold)

    for idx in range(len(layers)):

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()




def prune_magnitude_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Magnitude Pruning on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0)
            thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
            W_mask = (W_metric>=thresh)
            # compress(layer, W_mask, device)
            compress(layer, W_mask, model.hf_device_map[f"model.layers.{i}"])
