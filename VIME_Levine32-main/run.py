import pandas as pd
from vime_utils import perf_metric
from vime_semi import vime_semi
from vime_self import vime_self
from supervised_models import logit, xgb_model, mlp
from data_loader import load_data
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


def run(p_m, alpha, K, beta, label_no):

    # Experimental parameters
    model_sets = ['logit', 'xgboost', 'mlp']

    # # Hyper-parameters
    # p_m = 0.3
    # alpha = 2.0
    # K = 5
    # beta = 0.6
    # label_no = 0.1

    # Define output
    results2 = np.zeros([len(model_sets)+2])
    results3 = np.zeros([len(model_sets)+2])
    results4 = np.zeros([len(model_sets)+2])
    results5 = np.zeros([len(model_sets)+2])
    results6 = np.zeros([len(model_sets)+2])
    results7 = np.zeros([len(model_sets)+2])
    results8 = np.zeros([len(model_sets)+2])

    # Load data
    x_train, y_train, x_unlab, x_test, y_test = load_data(label_no)

    print(x_train.shape)
    print(y_train.shape)
    print(x_unlab.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Logistic regression
    y_test_hat = logit(x_train, y_train, x_test)
    results2[0] = perf_metric('acc', y_test, y_test_hat)
    results3[0] = perf_metric('f1', y_test, y_test_hat)
    results4[0] = perf_metric('precision', y_test, y_test_hat)
    results5[0] = perf_metric('rec', y_test, y_test_hat)
    results6[0] = perf_metric('roc', y_test, y_test_hat)

    # XGBoost
    # y_test_hat = xgb_model(x_train, y_train, x_test)
    # results[1] = perf_metric(metric, y_test, y_test_hat)

    # MLP
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100

    y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
    results2[2] = perf_metric('acc', y_test, y_test_hat)
    results3[2] = perf_metric('f1', y_test, y_test_hat)
    results4[2] = perf_metric('precision', y_test, y_test_hat)
    results5[2] = perf_metric('rec', y_test, y_test_hat)
    results6[2] = perf_metric('roc', y_test, y_test_hat)\

    # Train VIME-Self
    vime_self_parameters = dict()
    vime_self_parameters['batch_size'] = 128
    vime_self_parameters['epochs'] = 10
    vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)

    # Save encoder
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    file_name = './save_model/encoder_model.h5'

    vime_self_encoder.save(file_name)

    # Train VIME-Semi
    vime_semi_parameters = dict()
    vime_semi_parameters['hidden_dim'] = 100
    vime_semi_parameters['batch_size'] = 128
    vime_semi_parameters['iterations'] = 1000
    y_test_hat = vime_semi(x_train, y_train, x_unlab, x_test,
                           vime_semi_parameters, p_m, K, beta, file_name)

    # Test VIME
    results2[4] = perf_metric('acc', y_test, y_test_hat)
    results3[4] = perf_metric('f1', y_test, y_test_hat)
    results4[4] = perf_metric('precision', y_test, y_test_hat)
    results5[4] = perf_metric('rec', y_test, y_test_hat)
    results6[4] = perf_metric('roc', y_test, y_test_hat)

    print("========= accuracy ===========")

    for m_it in range(len(model_sets)):

        model_name = model_sets[m_it]

        print('Supervised Performance, Model Name: ' + model_name +
              ', Performance: ' + str(results2[m_it]))

    print('VIME-Self Performance: ' + str(results2[m_it+1]))

    print('VIME Performance: ' + str(results2[m_it+2]))

    print("========= f1 score ===========")

    for m_it in range(len(model_sets)):

        model_name = model_sets[m_it]

        print('Supervised Performance, Model Name: ' + model_name +
              ', Performance: ' + str(results3[m_it]))

    print('VIME-Self Performance: ' + str(results3[m_it+1]))

    print('VIME Performance: ' + str(results3[m_it+2]))

    print("========= Precision  ===========")

    for m_it in range(len(model_sets)):

        model_name = model_sets[m_it]

        print('Supervised Performance, Model Name: ' + model_name +
              ', Performance: ' + str(results4[m_it]))

    print('VIME-Self Performance: ' + str(results4[m_it+1]))

    print('VIME Performance: ' + str(results4[m_it+2]))

    print("========= Recall Score  ===========")

    for m_it in range(len(model_sets)):

        model_name = model_sets[m_it]

        print('Supervised Performance, Model Name: ' + model_name +
              ', Performance: ' + str(results5[m_it]))

    print('VIME-Self Performance: ' + str(results5[m_it+1]))

    print('VIME Performance: ' + str(results5[m_it+2]))

    print("========= AUROC ===========")

    for m_it in range(len(model_sets)):

        model_name = model_sets[m_it]

        print('Supervised Performance, Model Name: ' + model_name +
              ', Performance: ' + str(results6[m_it]))

    print('VIME-Self Performance: ' + str(results6[m_it+1]))

    print('VIME Performance: ' + str(results6[m_it+2]))

    return [results2, results3, results4, results5, results6, results7, results8]


if __name__ == '__main__':
    run(0.2, 1.0, 3, 0.6, 250)
    run(0.2, 1.0, 3, 0.8, 250)
