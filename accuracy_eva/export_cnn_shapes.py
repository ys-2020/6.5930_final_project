import argparse
import os
import pickle

import torch
import torchvision
from torch import nn


def make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def prune(module: nn.Module, ratio: float, divisor: int = 8):
    if isinstance(module, nn.Conv2d):
        new_module = nn.Conv2d(
            make_divisible(module.in_channels * ratio, divisor=divisor) if module.in_channels != 3 else 3,
            make_divisible(module.out_channels * ratio, divisor=divisor),
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        return new_module
    elif isinstance(module, nn.BatchNorm2d):
        new_module = nn.BatchNorm2d(
            make_divisible(module.num_features * ratio, divisor=divisor),
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        return new_module
    else:
        for n, m in module.named_children():
            setattr(module, n, prune(m, ratio=ratio, divisor=divisor))
        return module


def get_model(name: str) -> nn.Module:
    if name == "resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif name == "resnet50@0.9_8":
        model = torchvision.models.resnet50(pretrained=False)
        model = prune(model, 0.9, divisor=8)
        model.fc = nn.Linear(make_divisible(model.fc.in_features * 0.9, divisor=8), 10)
        return model
    elif name == "resnet50@0.8_8":
        model = torchvision.models.resnet50(pretrained=False)
        model = prune(model, 0.8, divisor=8)
        model.fc = nn.Linear(make_divisible(model.fc.in_features * 0.8, divisor=8), 10)
        return model
    elif name == "resnet50@0.7_8":
        model = torchvision.models.resnet50(pretrained=False)
        model = prune(model, 0.7, divisor=8)
        model.fc = nn.Linear(make_divisible(model.fc.in_features * 0.7, divisor=8), 10)
        return model
    elif name == "resnet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif name == "resnet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif name == "resnet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif name == "densenet121":
        model = torchvision.models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
        return model
    elif name == "densenet201":
        model = torchvision.models.densenet201(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
        return model
    elif name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model
    else:
        raise NotImplementedError(f"Model {name} is not supported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = get_model(args.model_name).to(args.device)
    model.eval()

    cnn_layers = []

    def hook(module: nn.Conv2d, inputs, output):
        input = inputs[0]
        w = output.size(3)
        h = output.size(2)
        c = input.size(1)
        n = input.size(0)
        m = output.size(1)
        s = module.kernel_size[0]
        r = module.kernel_size[1]
        wpad = module.padding[1]
        hpad = module.padding[0]
        wstride = module.stride[1]
        hstride = module.stride[0]
        cnn_layers.append((w, h, c, n, m, s, r, wpad, hpad, wstride, hstride))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook)
    with torch.no_grad():
        model(torch.zeros([1, 3, args.input_size, args.input_size], device=args.device))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(cnn_layers, f)


if __name__ == "__main__":
    main()
