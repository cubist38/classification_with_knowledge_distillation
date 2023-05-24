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
    
    parser.add_argument('--model', default='efficientnet_v2_l', type = str,
                        help="name of model to use")
    parser.add_argument('--lr', default=1e-3, type = float)
    parser.add_argument('--device', default = 'cuda:0', type = str)
    parser.add_argument('--batch-size', default = 16, type = int)
    parser.add_argument('--num-epochs', default = 1000, type = int)
    parser.add_argument('--data-root', default = './data', type = str)
    parser.add_argument('--step-eval-epoch', default = 10, type = int)
    parser.add_argument('--save-dir', default = './weights', type = str)
    parser.add_argument('--log-dir', default = './logs', type = str)
    parser.add_argument('--resume', default = None, type = str)

    return parser

def main(args):
    print(args)
    CLASS_TO_INDEX = class_to_index(os.path.join(args.data_root, 'train'))
    n_classes = len(CLASS_TO_INDEX)
    device = torch.device(args.device)
    if args.resume is not None:
        model = build_model(args.model, n_classes, pretrained = False)
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
    else:
        model = build_model(args.model, n_classes)
    transform = model.transform()
    dataset_train = CustomDataset(os.path.join(args.data_root, 'train'), transform = transform, mapping = CLASS_TO_INDEX)
    train_dataloader = get_dataloader(dataset_train, batch_size = args.batch_size)
    dataset_test = CustomDataset(os.path.join(args.data_root, 'test'), transform =  transform, mapping = CLASS_TO_INDEX)
    test_dataloader = get_dataloader(dataset_test, batch_size = args.batch_size, shuffle = False)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.to(device)
    min_loss = torch.inf
    best = None
    epochs = args.num_epochs
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    log_path = os.path.join(args.log_dir, 'log.txt')
    with open(log_path, 'w') as f:
        for epoch in range(epochs):
            # Training
            train_loss = train_one_epoch(model,
                                        train_dataloader, 
                                        criterion,
                                        optimizer, 
                                        device)
            scheduler.step()
            print('Epoch: {} - Train loss: {:.4f}'.format(epoch, train_loss))
            f.write('Epoch: {} - Train loss: {:.4f}\n'.format(epoch, train_loss))
            # Evaluation
            if epoch > 0 and epoch % args.step_eval_epoch == 0:
                eval_loss = eval(model,
                                test_dataloader,
                                criterion,
                                device)
                print('Epoch: {} - Eval loss: {:.4f}'.format(epoch, eval_loss))
                f.write('Epoch: {} - Eval loss: {:.4f}\n'.format(epoch, eval_loss))
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best = model.state_dict()
    
    save_path = os.path.join(args.save_dir, 'best.pth')
    torch.save(best, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training with Knowledge Distillation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)