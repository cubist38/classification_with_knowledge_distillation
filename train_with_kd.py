import torch
import os
import argparse
from knowledge_distillation import KnowledgeDistillationLoss
import torchvision
from torchvision import transforms
from dataset import *
from engine import *
import PIL
from models.model import build_model
 

CLASSES = {
    '0': 'banana', 
    '1': 'bed', 
    '2': 'bicycle', 
    '3': 'book', 
    '4': 'bottle', 
    '5': 'car', 
    '6': 'cat', 
    '7': 'clock', 
    '8': 'cup', 
    '9': 'dog', 
    '10': 'laptop', 
    '11': 'sofa'
}

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for Knowledge Distillation training', add_help=False)
    
    parser.add_argument('--backbone', default='mobilenet_v2', type = str,
                        help="name of backbone to use")
    parser.add_argument('--lr', default=1e-3, type = float)
    parser.add_argument('--teacher-weights', required = True, default = './weights/teacher_weights.pt', type = str)
    parser.add_argument('--device', default = 'cuda:0', type = str)
    parser.add_argument('--data-root', default = './data', type = str)
    parser.add_argument('--batch-size', default = 16, type = int)
    parser.add_argument('--num-epochs', default = 1000, type = int)
    parser.add_argument('--student-weights', default = './student_weights.pt')

    return parser    

def main(args):
    print(args)
    n_classes = len(CLASSES)
    device = torch.device(args.device)

    
    teacher_model = build_model('efficientnet-b4', num_classes = n_classes)
    image_size = teacher_model.image_size()
    checkpoint = torch.load(args.teacher_weights, map_location = torch.device('cpu'))
    teacher_model.load_state_dict(checkpoint)
    teacher_model.to(device)
    teacher_model.eval()
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = CustomDataset(os.path.join(args.data_root, 'train'), transform =  transform_train)
    train_dataloader = get_dataloader(dataset_train, batch_size = args.batch_size)
    transform_test = transforms.Compose([
        transforms.Resize(380, interpolation= PIL.Image.BICUBIC),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_test = CustomDataset(os.path.join(args.data_root, 'test'), transform =  transform_test)
    test_dataloader = get_dataloader(dataset_test, batch_size = args.batch_size)
    student_model = torchvision.models.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V2)
    student_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features = n_classes, bias=True),
    )
    optimizer = torch.optim.Adam(student_model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    student_model.to(device)
    min_loss = torch.inf
    weights = None
    epochs = args.num_epochs
    criterion = torch.nn.CrossEntropyLoss()
    kd = KnowledgeDistillationLoss(criterion, 10, 0.8)

    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch_kd(student_model,
                                        teacher_model,
                                        kd,
                                        train_dataloader, 
                                        optimizer, 
                                        device)
        scheduler.step()
        print('Epoch: {} - Train loss: {:.4f}'.format(epoch, train_loss))
        # Evaluation
        if epoch > 0:
            eval_loss = eval_kd(student_model,
                                criterion,
                                test_dataloader,
                                device)
            print('Epoch: {} - Eval loss: {:.4f}'.format(epoch, eval_loss))
            if eval_loss < min_loss:
                min_loss = eval_loss
                weights = student_model.state_dict()

    torch.save(weights, args.student_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training with Knowledge Distillation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)