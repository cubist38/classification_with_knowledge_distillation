import torch
import os
import argparse
from torchvision import transforms
from dataset import *
from engine import *
from PIL import Image
from models.model import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for Knowledge Distillation training', add_help=False)
    
    parser.add_argument('--model', default='efficientnet-b4', type = str,
                        help="name of model to use")
    parser.add_argument('--lr', default=1e-3, type = float)
    parser.add_argument('--device', default = 'cuda:0', type = str)
    parser.add_argument('--batch-size', default = 16, type = int)
    parser.add_argument('--num-epochs', default = 1000, type = int)
    parser.add_argument('--data-root', default = './data', type = str)

    return parser

def main(args):
    print(args)
    CLASS_TO_INDEX = class_to_index(os.path.join(args.data_root, 'train'))
    n_classes = len(CLASS_TO_INDEX)
    device = torch.device(args.device)

    model = build_model(args.model, n_classes)
    image_size = model.image_size()
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = CustomDataset(os.path.join(args.data_root, 'train'), transform = transform_train, mapping = CLASS_TO_INDEX)
    train_dataloader = get_dataloader(dataset_train, batch_size = args.batch_size)
    transform_test = transforms.Compose([
        transforms.Resize(380, interpolation= Image.BICUBIC),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_test = CustomDataset(os.path.join(args.data_root, 'test'), transform =  transform_test, mapping = CLASS_TO_INDEX)
    test_dataloader = get_dataloader(dataset_test, batch_size = args.batch_size, shuffle = False)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.to(device)
    min_loss = torch.inf
    weights = None
    epochs = args.num_epochs
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch(model,
                                    train_dataloader, 
                                    criterion,
                                    optimizer, 
                                    device)
        scheduler.step()
        print('Epoch: {} - Train loss: {:.4f}'.format(epoch, train_loss))
        # Evaluation
        if epoch > 0 and epoch % 5 == 0:
            eval_loss = eval(model,
                            test_dataloader,
                            criterion,
                            device)
            print('Epoch: {} - Eval loss: {:.4f}'.format(epoch, eval_loss))
            if eval_loss < min_loss:
                min_loss = eval_loss
                weights = model.state_dict()
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weights_path = os.path.join('./weights', 'best.pth')
    torch.save(weights, weights_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training with Knowledge Distillation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)