import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from dataset.dataset import DogCatDataset
from .base_task import BaseTask
from evaluation import accuracy
from torchvision import transforms
from torch.utils.data import DataLoader


class TrainingSimpleCNN(BaseTask):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self, train_dataloader):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with Cross Entropy Loss' % self.running_epoch, unit='it', total=len(train_dataloader)) as pbar:
            for it, items in enumerate(train_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                out = self.model(items['image'])
                    
                self.optimizer.zero_grad()
                
                loss = self.loss_fn(out, items['label'])
                loss.backward()

                self.optimizer.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
        self.scheduler.step()

    def evaluate_loss(self, dev_dataloader):
        self.model.eval()
        running_loss = 0
        with tqdm(desc='Epoch %d - Validation' % self.running_epoch, unit='it', total=len(dev_dataloader)) as pbar:
            for it, items in enumerate(dev_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                
                with torch.no_grad():   
                    out = self.model(items['image'])
                
                loss = self.loss_fn(out, items['label'])
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()      
        
    def evaluation(self, test_dataloader):
        gts = []
        gens = []
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.running_epoch, unit='it', total=len(test_dataloader)) as pbar:
            for it, items in enumerate(test_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items['image']).softmax(dim=-1)
                gt = items['label']
                gts.append(gt)
                gens.append(outs)

                pbar.update()
        gts = torch.stack(gts)
        gens = torch.stack(gens)
        
        acc = accuracy(gens, gts)
        scores = {
            'accuracy': acc
        }

        print(scores)
        return scores

    def load_datasets(self, config):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),       # keep aspect ratio
            transforms.CenterCrop(224),   # crop center
            transforms.ToTensor(),
        ])

        self.train_dataset = DogCatDataset(config.DATA.JSON_PATH.TRAIN,
                                           config.DATA,
                                           train_transform)
            
        self.test_dataset = DogCatDataset(config.DATA.JSON_PATH.TEST,
                                          config.DATA,
                                          test_transform)
        
        self.dev_dataset = DogCatDataset(config.DATA.JSON_PATH.DEV,
                                         config.DATA,
                                         test_transform)
        
    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            config.TRAINING.BATCH_SIZE,
            collate_fn=self.train_dataset.collate_fn
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate_fn
        )
        
        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            batch_size=1,
            collate_fn=self.dev_dataset.collate_fn
        )