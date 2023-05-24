import argparse
from dataset import *
import torch
from models.model import build_model
import PIL
from torchvision import transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for Knowledge Distillation training', add_help=False)
    
    parser.add_argument('--model', default='efficientnet_b4', type = str,
                        help="name of model to use")
    parser.add_argument('--device', default = 'cuda:0', type = str)
    parser.add_argument('--batch-size', default = 16, type = int)
    parser.add_argument('--data-root', default = './data', type = str)
    parser.add_argument('--weights', default = './weights/best.pth', type = str)

    return parser

def main(args):
    device = torch.device(args.device)
    CLASS_TO_INDEX = class_to_index(os.path.join(args.data_root, 'train'))
    n_classes = len(CLASS_TO_INDEX)
    model = build_model(model_name = args.model, n_classes = n_classes, pretrained = False)
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(380, interpolation= PIL.Image.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_test = CustomDataset(os.path.join(args.data_root, 'test'), transform =  transform, mapping = CLASS_TO_INDEX)
    n_samples = len(dataset_test)
    test_dataloader = get_dataloader(dataset_test, batch_size = args.batch_size, shuffle = False)
    total_true_predicted_samples = 0.0
    with torch.no_grad():
        for samples, targets in tqdm.tqdm(test_dataloader, total = len(test_dataloader)):
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            predicted_targets = torch.argmax(torch.softmax(outputs, dim = 1), dim = 1)
            total_true_predicted_samples +=  torch.sum(predicted_targets == targets).item()

    print('Accuracy: ', total_true_predicted_samples / n_samples)
        
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)