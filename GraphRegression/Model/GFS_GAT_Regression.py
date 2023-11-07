import os
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from scipy.special import softmax
from sklearn import preprocessing
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, mean_squared_error
from GraphClassification.Utils.PlotHelper import plot_acc, plot_loss
from GraphClassification.Utils.Preprocess import tensor_from_numpy, Preprocess_GandMUandY, GraphStruct2XandA
from GraphClassification.Utils.TrainHelper import mini_batches
from GraphClassification.Model.GAT_GraphClassifier import GraphClassifier


# @Time : 2023/02/04
# @Author : Fuping Hu
# @Email : hfping@stu.jiangnan.edu.cn

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
        # HyperParameters Settings
        self.EPOCHS = HyperParams['EPOCHS']
        self.HIDDEN_DIM = HyperParams['HIDDEN_DIM']
        self.LEARNING_RATE = HyperParams['LEARNING_RATE']
        self.WEIGHT_DECAY = HyperParams['WEIGHT_DECAY']
        self.PATIENCE = HyperParams['PATIENCE']

        # define loss function and optimizer
        self.criterion = None
        self.optimizer = None

        # for visual
        self.acc_list = list()
        self.loss_list = list()

        # define results store
        self.train_loss_list = list()
        self.train_rmse_list = list()

        self.val_loss_list = list()
        self.val_rmse_list = list()

        self.test_loss_list = list()
        self.test_rmse_list = list()

        self.test_rmse = 0

        # Early stopping model Parameters Saved Path
        Time = int(round(time.time() * 1000))
        TimeStr = time.strftime('%Y%m%d_%H%M%S', time.localtime(Time / 1000))
        saved_path = './TempModel/' + self.DatasetName
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        self.saved_path = './TempModel/{}/{}_GFS-GAT-Regression_{}_{}.pkl'.format(self.DatasetName, self.DatasetName, self.DEVICE,
                                                                       TimeStr)

    # Fitting the model
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

        # Create module list
        self.ModuleList = nn.ModuleList()
        for k in range(self.num_rules):
            self.ModuleList.append(
                GraphClassifier(input_dim=input_dim, hidden_dim=self.HIDDEN_DIM, num_classes=self.num_class).to(
                    self.DEVICE))

        # Cross entropy loss function
        self.criterion = nn.MSELoss().to(self.DEVICE)

        self.optimizer = optim.Adam(params=self.ModuleList.parameters(), lr=self.LEARNING_RATE,
                                    weight_decay=self.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        logger.info(self.ModuleList)

        # Build  models according to the number of rules
        self.mu4Val_Model = []
        for i in range(MuList_Val.size(1)):
            self.mu4Val_Model.append(MuList_Val[:, i].reshape(MuList_Val.size(0), 1))

        # Initialize the early_stopping object
        best_val_rmse, count, best_position = 100, 0, 0
        Epoch = list()
        for epoch in range(self.EPOCHS):
            # For plot acc figure
            Epoch.append(epoch + 1)

            # For Train with mini batch
            Count_NumBatch = 0
            self.batch_train_rmses = list()
            self.batch_train_losses = list()

            self.batch_train_rmse_mean = 0
            self.batch_train_loss_mean = 0

            self.batch_val_rmses = list()
            self.batch_val_losses = list()

            self.batch_val_rmse_mean = 0
            self.batch_val_loss_mean = 0

            # Training with mini-batch
            for minibatch in minibatches:
                Count_NumBatch += 1
                (minibatch_X, minibatch_Mu, minibatch_Y) = minibatch
                InputAdjHat, Ind, y_train, input_dim, X_all, MuListTrain = Preprocess_GandMUandY(minibatch_X,
                                                                                                 minibatch_Mu,
                                                                                                 minibatch_Y)
                X_all = torch.tensor(X_all, dtype=torch.float32).to(self.DEVICE)
                y_train = torch.DoubleTensor(y_train).to(self.DEVICE)

                self.ModuleList.train()
                self.optimizer.zero_grad()

                self.mu4Train_Model = []
                logits = torch.Tensor()
                for i in range(MuListTrain.size(1)):
                    self.mu4Train_Model.append(MuListTrain[:, i].reshape(MuListTrain.size(0), 1))
                    temp_train = self.mu4Train_Model[i] * self.ModuleList[i](InputAdjHat, X_all, Ind, y_train)
                    if i == 0:
                        logits = temp_train
                    else:
                        logits += temp_train

                loss = torch.sqrt(self.criterion(logits.to(self.DEVICE), y_train))
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()

                # Classification metrics
                train_rmse = mean_squared_error(y_train.cpu().detach().numpy(), logits.max(1)[1].cpu().detach().numpy(), squared=False)

                self.batch_train_losses.append(loss.item())
                self.batch_train_rmses.append(train_rmse)

                # For Val
                logits_val = torch.Tensor()
                for i in range(MuList_Val.size(1)):
                    temp_val = self.mu4Val_Model[i] * self.ModuleList[i](InputAdjHat_val, torch.tensor(X_all_val,
                                                                                                       dtype=torch.float32).to(
                        self.DEVICE), Ind_val, y_val)
                    if i == 0:
                        logits_val = temp_val
                    else:
                        logits_val += temp_val

                loss_val = torch.sqrt(self.criterion(logits_val.to(self.DEVICE), torch.DoubleTensor(y_val).to(self.DEVICE)))
                val_rmse = mean_squared_error(y_val, logits_val.max(1)[1].cpu().detach().numpy(), squared=False)

                self.batch_val_losses.append(loss_val.item())
                self.batch_val_rmses.append(val_rmse)
                logger.info('Epoch:{:05d}/{:05d}--Batch:{:03d}/{:03d}|Train Loss:{:.6f},Train RMSE: {:.4}|Val Loss:{:.6f},Val RMSE:{:.4}'.format(
                        epoch + 1, self.EPOCHS, Count_NumBatch, num_batch, loss.item(), train_rmse, loss_val.item(), val_rmse))

                if Count_NumBatch == num_batch:
                    self.batch_train_rmse_mean = np.mean(self.batch_train_rmses)
                    self.batch_train_loss_mean = np.mean(self.batch_train_losses)

                    self.batch_val_rmse_mean = np.mean(self.batch_val_rmses)
                    self.batch_val_loss_mean = np.mean(self.batch_val_losses)

                    self.train_rmse_list.append(self.batch_train_rmse_mean)
                    self.train_loss_list.append(self.batch_train_loss_mean)

                    self.val_rmse_list.append(self.batch_val_rmse_mean)
                    self.val_loss_list.append(self.batch_val_loss_mean)

            scheduler.step()

            logger.info('Current Learning Rate: {}'.format(scheduler.get_last_lr()))
            # Early stopping
            if self.batch_val_rmse_mean < best_val_rmse:
                best_val_rmse = self.batch_val_rmse_mean
                count = 0
                best_position = epoch + 1
                ModelDict = {}
                for k in range(self.num_rules):
                    ModelDict['Model{}'.format(k)] = self.ModuleList[k].state_dict()

                torch.save(ModelDict, self.saved_path)
                logger.info('Saving The Temp Best Val RMSE Model...')
            else:
                count += 1
                patience = self.PATIENCE
                if count > patience:
                    logger.info('Early Stopping Epoch:{}'.format(epoch + 1))
                    # plot_acc(self.LEARNING_RATE, self.WEIGHT_DECAY, Epoch, self.train_acc_list, self.val_acc_list,
                    #          self.DatasetName, self.num_rules, self.HIDDEN_DIM, Method_Name='GFS-GCN')
                    # plot_loss(self.LEARNING_RATE, self.WEIGHT_DECAY, Epoch, self.train_acc_list, self.val_acc_list,
                    #          self.DatasetName, self.num_rules, self.HIDDEN_DIM, Method_Name='GFS-GCN')
                    break
        # plot_acc(self.LEARNING_RATE, self.WEIGHT_DECAY, Epoch, self.train_acc_list, self.val_acc_list,
        #          self.DatasetName, self.num_rules, self.HIDDEN_DIM, Method_Name='GFS-GCN')
        # plot_loss(self.LEARNING_RATE, self.WEIGHT_DECAY, Epoch, self.train_loss_list, self.val_loss_list,
        #           self.DatasetName, self.num_rules, self.HIDDEN_DIM, Method_Name='GFS-GCN')

    def predict(self):
        # Compute the normal firing level
        self.mu4test_list = softmax(self.mu4test_list, axis=1)
        self.mu4test_list = tensor_from_numpy(self.mu4test_list, self.DEVICE)

        OriginTestData = self.X_test
        min_max_scaler = preprocessing.MinMaxScaler()
        for originTestData in OriginTestData:
            X_test, A_test = GraphStruct2XandA(originTestData[0], originTestData[1])
            X_test_normal = min_max_scaler.fit_transform(X_test)
            self.NodeFeatures_Test.append(X_test_normal)
            self.AdjMat_Test.append(A_test)

        # Adjacent matrix diagonalized splicing
        A_all = self.AdjMat_Test[0]
        for id, adj in enumerate(self.AdjMat_Test):
            if id != 0:
                A_all = scipy.linalg.block_diag(A_all, adj)

        # Splicing of node features
        # Construction of the node indicater
        X_all_test = list()
        Ind_test = list()
        for id, X in enumerate(self.NodeFeatures_Test):
            for X_feature in X:
                X_all_test.append(X_feature)
                Ind_test.append(id)
        X_all_test = np.array(X_all_test)
        Ind_test = np.array(Ind_test)

        # Add self-loop of Adj
        I = np.array(np.eye(A_all.shape[0]))
        A_all_hat = A_all + I
        # Compute degree matrix
        D_hat = np.sum(A_all_hat, axis=0)
        # Feature propagation
        D_hat = np.diag(D_hat ** -0.5)

        y_test = np.array(self.Y_test)
        InputAdjHat_test = (D_hat.dot(A_all_hat)).dot(D_hat)
        self.mu4Test_Model = []
        for i in range(self.mu4test_list.size(1)):
            self.mu4Test_Model.append(self.mu4test_list[:, i].reshape(self.mu4test_list.size(0), 1))

        logger.info('Loading Optiming Model Parameter File ,Predicting...')
        checkpoint = torch.load(self.saved_path)
        for k in range(self.num_rules):
            self.ModuleList[k].load_state_dict(checkpoint['Model{}'.format(k)])
        self.ModuleList.eval()

        with torch.no_grad():
            logits_test = torch.Tensor()
            for k in range(self.num_rules):
                temp_test = self.mu4Test_Model[k] * self.ModuleList[k](
                    torch.tensor(InputAdjHat_test, dtype=torch.float32).to(self.DEVICE),
                    torch.tensor(X_all_test, dtype=torch.float32).to(self.DEVICE), Ind_test, y_test)
                if k == 0:
                    logits_test = temp_test
                else:
                    logits_test += temp_test
            test_loss = torch.sqrt(self.criterion(logits_test.to(self.DEVICE), torch.DoubleTensor(y_test).to(self.DEVICE)))
            # test_acc = torch.eq(logits_test.max(1)[1], torch.tensor(y_test).to(self.DEVICE)).float().mean()
            test_rmse = mean_squared_error(y_test, logits_test.max(1)[1].cpu().detach().numpy(), squared=False)

            self.test_loss_list.append(test_loss.item())
            self.test_rmse_list.append(test_rmse)
            logger.info('Testing RMSE: {:.4}'.format(test_rmse))
            self.test_rmse = test_rmse

    def return_res(self):
        return self.test_rmse
