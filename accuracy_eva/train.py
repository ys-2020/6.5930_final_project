import argparse
import os

import torch
import torchvision
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange


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
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    exp_root = os.path.join("exps", args.exp_name)
    wandb.init(project="dl_arch", config=vars(args), name=args.exp_name)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    train_dataset = torchvision.datasets.cifar.CIFAR10(
        root="dataset",
        train=True,
        download=True,
        transform=transforms.Compose([transform, transforms.RandomHorizontalFlip()]),
    )
    test_dataset = torchvision.datasets.cifar.CIFAR10(root="dataset", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    os.makedirs(exp_root, exist_ok=True)

    model = get_model(args.model_name).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    step = 0
    pbar = trange(args.num_epochs, desc="Epochs", dynamic_ncols=True, leave=False, position=0)
    best_acc, best_epoch = 0, 0
    for epoch in pbar:
        model.train()
        for data, targets in tqdm(train_dataloader, desc="Training Set", dynamic_ncols=True, leave=False, position=1):
            data = data.cuda()
            targets = targets.cuda()
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                pbar.write(f"Step {step}, loss: {loss.item()}")
                wandb.log({"train_loss": loss.item(), "step": step}, step=step)
            step += 1

        model.eval()
        total_loss, num_correct = 0, 0
        cnt = 0
        with torch.no_grad():
            for data, targets in tqdm(test_dataloader, desc="Test Set", dynamic_ncols=True, leave=False, position=1):
                data = data.cuda()
                targets = targets.cuda()
                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item() * len(targets)
                num_correct += (outputs.argmax(1) == targets).sum().item()
                cnt += len(targets)
        test_loss = total_loss / cnt
        test_acc = num_correct / cnt
        wandb.log({"test_loss": test_loss, "test_acc": test_acc, "epoch": epoch}, step=step)
        pbar.write(f"###Epoch {epoch}: test_loss: {test_loss}, test_acc: {test_acc}")

        torch.save(model.state_dict(), os.path.join(exp_root, "latest.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(exp_root, "best.pth"))
        scheduler.step()


if __name__ == "__main__":
    main()
