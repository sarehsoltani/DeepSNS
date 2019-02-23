import torch.nn.functional as F
import torch
from tqdm import tqdm

class EEG:

    def __init__(self):
        pass

    def train(self, model, data_loader, optimizer, epoch):  
        
        model.train()

        tq = tqdm(data_loader, desc='{} E{}'.format('Train', str(epoch)))
        for step, (data, labels) in enumerate(tq):

            # zero all previous gradients
            optimizer.zero_grad()

            # get model outputs
            preds = model(data)

            # compute error
            loss = F.binary_cross_entropy_with_logits(preds, labels)

            # compute gradients and update model weights
            loss.backward()
            optimizer.step()

            # TODO: compute accuracy

            # update tqdm
            tq.set_description(desc='Loss {}'.format(loss))

    def evaluate(self, model, data_loader, epoch):
        model.eval()

        # disable gradients
        with torch.no_grad():
            tq = tqdm(data_loader, desc='{} E{}'.format('Validation', str(epoch)))
            for step, (data, labels) in enumerate(tq):
                
                # get model outputs
                preds = model(data)

                # compute error
                loss = F.nll_loss(preds, labels)

                # TODO: compute accuracy

                # update tqdm
                tq.set_description(desc='Loss {}'.format(loss))