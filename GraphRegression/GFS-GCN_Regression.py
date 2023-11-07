import json
import os
import time

import numpy
import numpy as np
import pandas
from loguru import logger
from sklearn.model_selection import KFold, train_test_split

from Model.GFS_GCN_Regression import GraphFuzzySystem
from Utils.GetAntecedent import GetAntWithVal
from Utils.ReadData import read_regression_data

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    HyperParams_json = open('./Config/hyperparams.json')
    HyperParamConfig = json.load(HyperParams_json)
    DatasetName = 'freesolv'
    DataContent = read_regression_data(DatasetName, prefer_attr_nodes=True)
    G, y = DataContent.data, DataContent.target
    y_len = len(y)
    y = numpy.array(y).reshape(y_len, 1)
    num_class = y.shape[1]
    Time = int(round(time.time() * 1000))
    TimeStr = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(Time / 1000))
    logger.add(
        "./results/logs/{}/GFS_GCN_Regression_Dataset-{}_TimeStamp-{}.log".format(DatasetName, DatasetName, TimeStr))
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    Epochs = HyperParamConfig['Epochs']
    best_result = dict()
    best_result['Epochs'] = Epochs
    best_result['best_rmse'] = 0
    # best_result['best_acc'] = 0
    best_result['num_rules'] = 0
    best_result['l2'] = 0
    best_result['lr'] = 0
    best_result['HiddenDim'] = 0
    best_result['Mini_Batch_Size'] = 0
    best_result['best_rmse_std'] = 0
    all_result = list()

    Rules = HyperParamConfig['Rules']
    for rule in Rules:
        temp_res_list = dict()
        num_rules = rule
        temp_res_list['Epochs'] = Epochs
        temp_res_list['num_rules'] = num_rules
        L2s = HyperParamConfig['L2s']
        for l2 in L2s:
            temp_res_list['l2'] = l2
            Lrs = HyperParamConfig['Lrs']
            for lr in Lrs:
                temp_res_list['lr'] = lr
                HiddenDims = HyperParamConfig['HiddenDims']
                for HiddenDim in HiddenDims:
                    temp_res_list['HiddenDim'] = HiddenDim
                    MiniBatchSize = HyperParamConfig['MiniBatchSize']
                    for mini_batch_size in MiniBatchSize:
                        temp_res_list['mini_batch_size'] = mini_batch_size
                        test_rmses = list()
                        num_kf = 0
                        for train_index, test_index in kf.split(G, y):
                            num_kf += 1
                            logger.info('Dataset:{}'.format(DatasetName))
                            logger.info('{}-Fold Cross Validation: {}/{}'.format(num_folds, num_kf, num_folds))

                            G_train, y_train = np.array(G)[train_index], np.array(y)[train_index]
                            G_test, y_test = np.array(G)[test_index], np.array(y)[test_index]
                            G_train_tra, G_train_val, y_train_tra, y_train_val = train_test_split(G_train, y_train,
                                                                                                  test_size=0.25,
                                                                                                  shuffle=True)
                            # HyperParams:
                            HyperParams = dict()
                            HyperParams['RULES'] = num_rules
                            HyperParams['EPOCHS'] = Epochs
                            HyperParams['HIDDEN_DIM'] = HiddenDim
                            HyperParams['LEARNING_RATE'] = lr
                            HyperParams['PATIENCE'] = HyperParamConfig['Patience']
                            HyperParams['WEIGHT_DECAY'] = l2
                            HyperParams['Mini_Batch_Size'] = mini_batch_size
                            logger.info('HyperParam Settings: {}'.format(HyperParams))

                            # Get the prototype graph
                            ClusterCenters, Mu4Train_list, Mu4Val_list, Mu4Test_list = GetAntWithVal(
                                G_train_tra.tolist(), G_train_val.tolist(), G_test, num_rules)

                            # Build the GraphFuzzySystem model
                            GfsModel = GraphFuzzySystem(_X_train=G_train_tra, _Y_train=y_train_tra,
                                                        _X_val=G_train_val, _Y_val=y_train_val,
                                                        _X_test=G_test, _Y_test=y_test,
                                                        _num_rules=num_rules, _centers=ClusterCenters,
                                                        _num_class=num_class, mu4Train_list=Mu4Train_list,
                                                        mu4Val_list=Mu4Val_list, mu4Test_list=Mu4Test_list,
                                                        DatasetName=DatasetName, HyperParams=HyperParams,
                                                        mini_batch_size=mini_batch_size)
                            GfsModel.fit()
                            GfsModel.predict()
                            test_rmse = GfsModel.return_res()
                            logger.info('Test_Params:{}'.format(HyperParams))
                            logger.info('Test_RMSE:{}'.format(test_rmse))

                            test_rmses.append(test_rmse)
                        test_rmse_mean = np.mean(test_rmses)
                        test_rmse_std = np.std(test_rmses)
                        if best_result['best_rmse'] == 0 & best_result['num_rules'] == 0 & best_result[
                            'l2'] == 0 & best_result['lr'] == 0 & best_result['HiddenDim'] == 0 & best_result['Mini_Batch_Size'] == 0 & best_result['best_rmse_std'] == 0:
                            best_result['best_rmse'] = test_rmse_mean
                            best_result['num_rules'] = num_rules
                            best_result['l2'] = l2
                            best_result['lr'] = lr
                            best_result['HiddenDim'] = HiddenDim
                            best_result['Mini_Batch_Size'] = mini_batch_size
                            best_result['best_rmse_std'] = test_rmse_std
                        if test_rmse_mean < best_result['best_rmse']:
                            best_result['best_rmse'] = test_rmse_mean
                            best_result['num_rules'] = num_rules
                            best_result['l2'] = l2
                            best_result['lr'] = lr
                            best_result['HiddenDim'] = HiddenDim
                            best_result['Mini_Batch_Size'] = mini_batch_size
                            best_result['best_rmse_std'] = test_rmse_std
                            logger.info('The temp best_result:{}'.format(best_result))
                        temp_res_list['test_rmse_mean'] = test_rmse_mean
                        temp_res_list['test_rmse_std'] = test_rmse_std

                        # Store Results into CSV
                        file_name = os.path.basename(__file__)
                        file_name = file_name.split('.')[0]
                        saved_dir = './results/csv/{}'.format(DatasetName)
                        if not os.path.exists(saved_dir):
                            os.makedirs(saved_dir)
                        pandas.DataFrame([best_result]).to_csv('./results/csv/{}/{}_{}.csv'.format(DatasetName,  file_name, TimeStr),
                                                                 index=False, mode='a')

                        logger.info(best_result)
                        all_result.append(best_result)
