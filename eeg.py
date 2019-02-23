import torch.nn.functional as F
import torch
from torch.autograd import Variable
from tqdm import tqdm

class EEG:

    def __init__(self):

        self.train_iters = 0
        self.val_iters = 0

    def train(self, model, data_loader, optimizer, writer, epoch):
        model.train()

        tq = tqdm(data_loader, desc='{} E{}'.format('Train', str(epoch)))
        for step, (data, labels) in enumerate(tq):
            
            # update train iters
            self.train_iters += 1

            # zero all previous gradients
            optimizer.zero_grad()

            # variable options
            var_params = {
                'requires_grad': False
            }

            # transfer data to device
            data = Variable(data.cuda(async=True), **var_params)
            labels = Variable(labels.cuda(async=True), **var_params)

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

            # write scalar
            writer.add_scalar('/loss', loss.data.item(), self.train_iters)

            
    def evaluate(self, model, data_loader, writer, epoch):
        model.eval()

        # disable gradients
        with torch.no_grad():

            tq = tqdm(data_loader, desc='{} E{}'.format('Validation', str(epoch)))
            for step, (data, labels) in enumerate(tq):
                
                # update train iters
                self.val_iters += 1

                # get model outputs
                preds = model(data)

                # compute error
                loss = F.nll_loss(preds, labels)

                # TODO: compute accuracy

                # update tqdm
                tq.set_description(desc='Loss {}'.format(loss))

                # write scalar
                writer.add_scalar('/loss', loss.data.item(), self.val_iters)