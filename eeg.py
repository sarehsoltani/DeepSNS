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
            loss = F.binary_cross_entropy_with_logits(preds, labels)

            # compute gradients and update model weights
            loss.backward()
            optimizer.step()

            # TODO: compute accuracy
            batch_score = float(self.evaluate(predicted=preds, Y=labels, params=["auc"])[0])

            # add new scores and losses to the list
            accs.append(batch_score)
            lsss.append(loss.data.item())

            mean_loss = sum(lsss) / float(len(lsss))
            mean_acc = sum(accs) / float(len(accs))

            # update tqdm
            tq.set_description(desc='| Loss: {}, AUC: {} |'.format(mean_loss, mean_acc))

            # write scalar
            writer.add_scalar('/loss', mean_loss, self.train_iters)
            writer.add_scalar('/auc-score', mean_acc, self.train_iters)

            
    def validate(self, model, data_loader, writer, epoch):
        model.eval()

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
                loss = F.binary_cross_entropy_with_logits(preds, labels)

                # TODO: compute accuracy
                score = float(self.evaluate(predicted=preds, Y=labels, params=["auc"])[0])

                accs.append(score)
                lsss.append(loss.data.item())

                mean_acc = sum(accs) / len(accs)
                mean_loss =  sum(lsss) / len(lsss)

                # update tqdm
                tq.set_description(desc='| Loss: {}, Accuracy: {} |'.format(mean_loss, mean_acc))

                # write scalar
                writer.add_scalar('/loss', mean_acc, self.val_iters)
                writer.add_scalar('/auc-score', mean_loss, self.val_iters)


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