import os
import time
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn import preprocessing
import scipy
import numpy as np

from GraphClassification.Utils.Preprocess import tensor_from_numpy, Preprocess_GandMUandY, GraphStruct2XandA
from GraphClassification.Utils.TrainHelper import mini_batches
from GraphClassification.Utils.PlotHelper import plot_acc, plot_loss

class GraphFuzzySystem(nn.Module):
    def __init__(self, _X_train: list, _Y_train: list, _X_val: list, _Y_val: list, _X_test: list, _Y_test: list,
                 _num_rules: int, _centers: list,
                 _num_class: int, mu4Train_list,
                 mu4Val_list, mu4Test_list,
                 DatasetName, HyperParams: dict,
                 mini_batch_size):
        super(GraphFuzzySystem, self).__init__()
        self.mini_batch_size = mini_batch_size
        self.X_train = _X_train
        self.Y_train = _Y_train
        self.X_val = _X_val
        self.Y_val = _Y_val
        self.X_test = _X_test
        self.Y_test = _Y_test
        self.num_rules = _num_rules
        self.centers = _centers
        self.num_class = _num_class
        self.graph_process_units = None
        self.DatasetName = DatasetName

        self.mu4train_list = mu4Train_list
        self.mu4val_list = mu4Val_list
        self.mu4test_list = mu4Test_list

        self.NodeFeatures_Train = list()
        self.AdjMat_Train = list()

        self.NodeFeatures_Val = list()
        self.AdjMat_Val = list()

        self.NodeFeatures_Test = list()
        self.AdjMat_Test = list()

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.EPOCHS = HyperParams['EPOCHS']
        self.HIDDEN_DIM = HyperParams['HIDDEN_DIM']
        self.LEARNING_RATE = HyperParams['LEARNING_RATE']
        self.WEIGHT_DECAY = HyperParams['WEIGHT_DECAY']
        self.PATIENCE = HyperParams['PATIENCE']
        self.criterion = None
        self.optimizer = None

        self.acc_list = list()
        self.loss_list = list()

        self.train_loss_list = list()
        self.train_acc_list = list()

        self.val_loss_list = list()
        self.val_acc_list = list()

        self.test_loss_list = list()
        self.test_acc_list = list()

        self.test_acc = 0

        Time = int(round(time.time() * 1000))
        TimeStr = time.strftime('%Y%m%d_%H%M%S', time.localtime(Time / 1000))
        saved_path = '../TempModel/' + self.DatasetName
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        self.saved_path = '../TempModel/{}/{}_GFS-GAT_{}_{}.pkl'.format(self.DatasetName, self.DatasetName, self.DEVICE,
                                                                TimeStr)

    def fit(self):
        X_4MB = self.X_train
        Mu_4MB = self.mu4train_list
        Y_4MB = self.Y_train

        self.mu4train_list = softmax(self.mu4train_list, axis=1)
        self.mu4train_list = tensor_from_numpy(self.mu4train_list, self.DEVICE)
        seed = 0
        minibatches, num_batch = mini_batches(X_4MB, Mu_4MB, Y_4MB, self.mini_batch_size, seed)

        Orign_X_Val = self.X_val
        Orign_mu4val_list = self.mu4val_list
        Orign_Y_val = self.Y_val
        InputAdjHat_val, Ind_val, y_val, input_dim, X_all_val, MuList_Val = Preprocess_GandMUandY(Orign_X_Val,
                                                                                                  Orign_mu4val_list,
                                                                                                  Orign_Y_val)

        for k in range(self.num_rules):
            exec('self.Model{} = GraphClassifier(input_dim=input_dim, hidden_dim=self.HIDDEN_DIM,'
                 'num_classes=self.num_class).to(self.DEVICE)'.format(k))

        self.criterion = nn.CrossEntropyLoss().to(self.DEVICE)

        params_list = []
        for k in range(self.num_rules):
            exec("temp_param = dict()")
            exec("temp_param['params'] = self.Model{}.parameters()".format(k))
            exec("params_list.append(temp_param)")
        self.optimizer = optim.Adam(params=params_list, lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        for k in range(self.num_rules):
            exec("logger.info(self.Model{})".format(k))

        for i in range(MuList_Val.size(1)):
            exec("mu4Val_Model{} = MuList_Val[:, {}].reshape(MuList_Val.size(0),1)".format(i, i))

        best_val_acc, count, best_position = 0, 0, 0
        Epoch = list()
        for epoch in range(self.EPOCHS):
            Epoch.append(epoch + 1)

            Count_NumBatch = 0
            self.batch_train_accs = list()
            self.batch_train_losses = list()
            self.batch_train_acc_mean = 0
            self.batch_train_loss_mean = 0

            self.batch_val_accs = list()
            self.batch_val_losses = list()
            self.batch_val_acc_mean = 0
            self.batch_val_loss_mean = 0

            for minibatch in minibatches:
                Count_NumBatch += 1
                (minibatch_X, minibatch_Mu, minibatch_Y) = minibatch
                InputAdjHat, Ind, y_train, input_dim, X_all, MuListTrain = Preprocess_GandMUandY(minibatch_X,
                                                                                                 minibatch_Mu,
                                                                                                 minibatch_Y)
                X_all = torch.tensor(X_all, dtype=torch.float32).to(self.DEVICE)
                y_train = torch.LongTensor(y_train).to(self.DEVICE)
                for k in range(self.num_rules):
                    exec("self.Model{}.train()".format(k))
                self.optimizer.zero_grad()

                for i in range(MuListTrain.size(1)):
                    exec("mu4Train_Model{} = MuListTrain[:, {}].reshape(MuListTrain.size(0),1)".format(i, i))

                for k in range(self.num_rules):
                    exec(
                        "logits{} = self.Model{}(InputAdjHat, X_all , Ind, y_train)".format(
                            k, k))
                    if k == 0:
                        exec("logits = logits{} * mu4Train_Model{}".format(k, k))
                    else:
                        exec("logits += logits{} * mu4Train_Model{}".format(k, k))
                exec("loss = self.criterion(logits.to(self.DEVICE), y_train)")
                exec("loss.requires_grad_(True)")
                exec("loss.backward()")
                self.optimizer.step()
                exec("train_acc = torch.eq(logits.max(1)[1], y_train).float().mean()")
                exec("self.batch_train_losses.append(loss.item())")
                exec("self.batch_train_accs.append(train_acc.item())")

                for k in range(self.num_rules):
                    exec(
                        "logits{}_val = self.Model{}(InputAdjHat_val,torch.tensor(X_all_val, dtype=torch.float32).to(self.DEVICE), Ind_val, y_val)".format(
                            k, k))
                    if k == 0:
                        exec("logits_val = logits{}_val * mu4Val_Model{}".format(k, k))
                    else:
                        exec("logits_val += logits{}_val * mu4Val_Model{}".format(k, k))
                exec("loss_val = self.criterion(logits_val.to(self.DEVICE), torch.LongTensor(y_val).to(self.DEVICE))")
                exec("val_acc = torch.eq(logits_val.max(1)[1], torch.tensor(y_val).to(self.DEVICE)).float().mean()")
                exec("self.batch_val_losses.append(loss_val.item())")
                exec("self.batch_val_accs.append(val_acc.item())")
                exec(
                    "logger.info('Epoch:{:05d}/{:05d}-Batch:{:03d}/{:03d}|Train Loss:{:.6f},TrainAcc: {:.4}|ValLoss:{:.6f},ValAcc:{:.4}'.format(epoch + 1, self.EPOCHS, Count_NumBatch, num_batch, loss.item(), train_acc.item(), loss_val.item(), val_acc.item()))")

                if Count_NumBatch == num_batch:
                    exec("self.batch_train_acc_mean = np.mean(self.batch_train_accs)")
                    exec("self.batch_train_loss_mean = np.mean(self.batch_train_losses)")
                    exec("self.batch_val_acc_mean = np.mean(self.batch_val_accs)")
                    exec("self.batch_val_loss_mean = np.mean(self.batch_val_losses)")

                    exec("self.train_acc_list.append(self.batch_train_acc_mean)")
                    exec("self.train_loss_list.append(self.batch_train_loss_mean)")
                    exec("self.val_acc_list.append(self.batch_val_acc_mean)")
                    exec("self.val_loss_list.append(self.batch_val_loss_mean)")

                    exec("temp_acc_list = [self.batch_train_acc_mean, self.batch_val_acc_mean]")
                    exec("temp_loss_list = [self.batch_train_loss_mean, self.batch_val_loss_mean]")
                    exec("self.acc_list.append(temp_acc_list)")
                    exec("self.loss_list.append(temp_loss_list)")
            scheduler.step()
            logger.info('Current Learning Rate: {}'.format(scheduler.get_last_lr()))
            if self.batch_val_acc_mean > best_val_acc:
                best_val_acc = self.batch_val_acc_mean
                count = 0
                best_position = epoch + 1
                ModelDict = {}
                for k in range(self.num_rules):
                    exec("ModelDict['Model{}'] = self.Model{}.state_dict()".format(k, k))
                torch.save(ModelDict, self.saved_path)
                logger.info('Saving The Temp Best Val_Acc Model...')
            else:
                count += 1
                patience = self.PATIENCE
                if count > patience:
                    logger.info('Early Stopping Epoch:{}'.format(epoch + 1))
                    break
        plot_acc(self.LEARNING_RATE, self.WEIGHT_DECAY, Epoch, self.train_acc_list, self.val_acc_list,
                 self.DatasetName, self.num_rules, self.HIDDEN_DIM, Method_Name='GFS-GAT')
        plot_loss(self.LEARNING_RATE, self.WEIGHT_DECAY, Epoch, self.train_loss_list, self.val_loss_list,
                  self.DatasetName, self.num_rules, self.HIDDEN_DIM, Method_Name='GFS-GAT')
    def predict(self):
        self.mu4test_list = softmax(self.mu4test_list, axis=1)
        self.mu4test_list = tensor_from_numpy(self.mu4test_list, self.DEVICE)

        OriginTestData = self.X_test
        min_max_scaler = preprocessing.MinMaxScaler()
        for originTestData in OriginTestData:
            X_test, A_test = GraphStruct2XandA(originTestData[0], originTestData[1])
            X_test_normal = min_max_scaler.fit_transform(X_test)
            self.NodeFeatures_Test.append(X_test_normal)
            self.AdjMat_Test.append(A_test)

        A_all = self.AdjMat_Test[0]
        for id, adj in enumerate(self.AdjMat_Test):
            if id != 0:
                A_all = scipy.linalg.block_diag(A_all, adj)

        X_all_test = list()
        Ind_test = list()
        for id, X in enumerate(self.NodeFeatures_Test):
            for X_feature in X:
                X_all_test.append(X_feature)
                Ind_test.append(id)
        X_all_test = np.array(X_all_test)
        Ind_test = np.array(Ind_test)

        I = np.array(np.eye(A_all.shape[0]))
        A_all_hat = A_all + I
        D_hat = np.sum(A_all_hat, axis=0)
        D_hat = np.diag(D_hat ** -0.5)

        y_test = np.array(self.Y_test)
        InputAdjHat_test = (D_hat.dot(A_all_hat)).dot(D_hat)

        for i in range(self.mu4test_list.size(1)):
            exec("mu4Test_Model{} = self.mu4test_list[:, {}].reshape(self.mu4test_list.size(0),1)".format(i, i))

        logger.info('Loading Model File ,Predicting...')
        checkpoint = torch.load(self.saved_path)
        for k in range(self.num_rules):
            exec("self.Model{}.load_state_dict(checkpoint['Model{}'])".format(k, k))

        for k in range(self.num_rules):
            exec("self.Model{}.eval()".format(k))

        with torch.no_grad():
            for k in range(self.num_rules):
                exec("logits{}_Test = self.Model{}(torch.tensor(InputAdjHat_test, dtype=torch.float32).to(self.DEVICE),"
                     "torch.tensor(X_all_test, dtype=torch.float32).to(self.DEVICE), Ind_test, y_test)".format(k, k))
                if k == 0:
                    exec("logits = logits{}_Test * mu4Test_Model{}".format(k, k))
                else:
                    exec("logits += logits{}_Test * mu4Test_Model{}".format(k, k))
            exec("test_loss = self.criterion(logits.to(self.DEVICE), torch.LongTensor(y_test).to(self.DEVICE))")
            exec("test_acc = torch.eq( logits.max(1)[1], torch.tensor(y_test).to(self.DEVICE)).float().mean()")
            exec("self.train_loss_list.append(test_loss.item())")
            exec("self.train_acc_list.append(test_acc.item())")
            exec("logger.info('Test Acc: {:.4}, Test: Loss {:.6f}'.format(test_acc.item(), test_loss.item()))")
            exec("self.test_acc =  test_acc.item()")

    def return_res(self):
        return self.test_acc
