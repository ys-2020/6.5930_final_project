import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm


def load_model(model_name, device="cuda"):
    model = torchvision.models.__dict__[model_name](pretrained=True)
    model.eval().to(device)
    return model


def eval_func(name, dataset):
    # Load the pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(name, device)

    # Set up eval dataset
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_dataset = ImageNet(root=dataset, split="val", transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="Running the evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print("=" * 50)
    print(f"Model: {name}")
    print(f"Accuracy on the ImageNet validation set: {accuracy:.2f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-list", type=str, default="alexnet,resnet18", help="list of models to evaluate, separated by commas"
    )
    parser.add_argument("--imagenet-path", type=str, default="/data/imagenet", help="path to the ImageNet dataset")
    args = parser.parse_args()

    imagenet_path = args.imagenet_path
    model_list = args.model_list.split(",")

    for model_name in model_list:
        eval_func(model_name, imagenet_path)


if __name__ == "__main__":
    main()
