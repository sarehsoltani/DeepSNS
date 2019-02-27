import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from torch.autograd import Variable
from tqdm import tqdm

class EEG:

    def __init__(self):

        self.train_iters = 0
        self.val_iters = 0

    def train(self, model, data_loader, tracker, optimizer, writer, epoch):
        model.train()
        
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

        loss_tracker = tracker.track('{}_loss'.format('train'), tracker_class(**tracker_params))
        acc_tracker = tracker.track('{}_acc'.format('train'), tracker_class(**tracker_params))

        accs = []
        lsss = []

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
            data = data.cuda()
            labels = labels.cuda()

            # data = Variable(data.cuda(async=True), **var_params)
            # labels = Variable(labels.cuda(async=True), **var_params)

            # get model outputs
            preds = model(data)

            # compute error
            BCE = nn.BCELoss()
            loss = BCE(preds, labels)

            # compute gradients and update model weights
            loss.backward()
            optimizer.step()

            # TODO: compute accuracy
            batch_score = float(self.evaluate(predicted=preds, Y=labels, params=["auc"])[0])
            loss_tracker.append(loss.data.item())
            acc_tracker.append(batch_score)

            # update tqdm
            tq.set_description(desc='| Train {} | Loss: {:10.5f}, AUC: {:10.5f} |'.format(epoch, loss_tracker.mean.value, acc_tracker.mean.value))

            # write scalar
            writer.add_scalar('/loss', loss_tracker.mean.value, self.train_iters)
            writer.add_scalar('/auc-score', acc_tracker.mean.value, self.train_iters)

            
    def validate(self, model, data_loader, tracker, writer, epoch):
        model.eval()

        tracker_class, tracker_params = tracker.MeanMonitor, {}

        loss_tracker = tracker.track('{}_loss'.format('validation'), tracker_class(**tracker_params))
        acc_tracker = tracker.track('{}_acc'.format('validation'), tracker_class(**tracker_params))

        # disable gradients
        with torch.no_grad():

            accs = []
            lsss = []

            tq = tqdm(data_loader, desc='{} E{}'.format('Validation', str(epoch)))

            for step, (data, labels) in enumerate(tq):

                # update train iters
                self.val_iters += 1

                # transfer data to device
                data = data.cuda()
                labels = labels.cuda()

                # get model outputs
                preds = model(data)

                # compute error
                BCE = nn.BCELoss()
                loss = BCE(preds, labels)

                # TODO: compute accuracy
                batch_score = float(self.evaluate(predicted=preds, Y=labels, params=["auc"])[0])

                loss_tracker.append(loss.data.item())
                acc_tracker.append(batch_score)

                # update tqdm
                tq.set_description(desc='| Val {} | Loss: {:10.5f}, AUC: {:10.5f} |'.format(epoch, loss_tracker.mean.value, acc_tracker.mean.value))

                # write scalar
                writer.add_scalar('/loss', acc_tracker.mean.value, self.val_iters)
                writer.add_scalar('/auc-score', loss_tracker.mean.value, self.val_iters)


    def evaluate(self, predicted, Y, params = ["auc"]):
        
        results = []
        predicted = predicted.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()

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