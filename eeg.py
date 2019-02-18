import torch.nn.functional as F
import torch

class EEG:

    def __init__(self):
        pass

    def train(self, model, data_loader, optimizer, epoch):  
        
        model.train()
        for _, (data, labels) in enumerate(data_loader):

            # zero all previous gradients
            optimizer.zero_grad()

            # get model outputs
            preds = model(data)

            # compute error
            loss = F.nll_loss(preds, labels)

            # compute gradients and update model weights
            loss.backward()
            optimizer.step()

            # TODO: compute accuracy


    def evaluate(self, model, data_loader, epoch):
        model.eval()

        # disable gradients
        with torch.no_grad():
            for _, (data, labels) in enumerate(data_loader):
                
                # get model outputs
                preds = model(data)

                # compute error
                loss = F.nll_loss(preds, labels)

                # TODO: compute accuracy
