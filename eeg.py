import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
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
            score = 100

            # update tqdm
            tq.set_description(desc='Loss: {}, Accuracy: {}'.format(loss, score))

            # write scalar
            writer.add_scalar('/loss', loss.data.item(), self.train_iters)

            
    def validate(self, model, data_loader, writer, epoch):
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
                score = 10

                # update tqdm
                tq.set_description(desc='Loss: {}, Accuracy: {}'.format(loss, score))

                # write scalar
                writer.add_scalar('/loss', loss.data.item(), self.val_iters)


    def evaluate(self, model, X, Y, params = ["acc"]):
        results = []
        batch_size = config.BATCH_SIZE
        
        predicted = []
        
        for i in range(len(X)/batch_size):
            s = i*batch_size
            e = i*batch_size+batch_size
            
            inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
            pred = model(inputs)
            
            predicted.append(pred.data.cpu().numpy())
            
            
        inputs = Variable(torch.from_numpy(X).cuda(0))
        predicted = model(inputs)
        
        predicted = predicted.data.cpu().numpy()
        
        for param in params:
            if param == 'acc':
                results.append(accuracy_score(Y, np.round(predicted)))
            if param == "auc":
                results.append(roc_auc_score(Y, predicted))
            if param == "recall":
                results.append(recall_score(Y, np.round(predicted)))
            if param == "precision":
                results.append(precision_score(Y, np.round(predicted)))
            if param == "fmeasure":
                precision = precision_score(Y, np.round(predicted))
                recall = recall_score(Y, np.round(predicted))
                results.append(2*precision*recall/ (precision+recall))
        return results