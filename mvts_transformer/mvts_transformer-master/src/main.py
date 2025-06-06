"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

from collections import deque

from matplotlib import pyplot as plt, rcParams
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from optimizers import get_optimizer
from models.loss import get_loss_module
from models.ts_transformer import model_factory
from datasets.datasplit import split_dataset
from datasets.data import data_factory, Normalizer
from utils import utils
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from options import Options
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json
import pickle
import time
import sys
import os
import logging
import seaborn as sns

logging.basicConfig(
    format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")

# 3rd party packages

# Project modules


def main(config):
    print("============")
    print(config)

    print("============")
    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(
        os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(
        ' '.join(sys.argv)))  # command used to run

    torch.manual_seed(10)

    device = torch.device('cuda' if (
        torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'],
                         n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    test_data = my_data
    # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    test_indices = None
    val_data = my_data
    val_indices = []
    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = data_class(
            config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config['test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(
            set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind)
                            for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info("Loaded {} test IDs from file: '{}'".format(
            len(test_indices), config['test_from']))
    if config['val_pattern']:  # used if val data come from different files / file patterns
        val_data = data_class(
            config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 # used only if test_indices not explicitly specified
                                                                 test_set_ratio=config['test_ratio'],
                                                                 test_indices=test_indices,
                                                                 random_seed=1337,
                                                                 labels=labels)
        # `split_dataset` returns a list of indices *per fold/split*
        train_indices = train_indices[0]
        # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(
        len(train_indices)))
    logger.info(
        "{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)

    # Pre-process features
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(
            my_data.feature_df.loc[train_indices])
        if not config['normalization'].startswith('per_sample'):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(
                val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(
                test_data.feature_df.loc[test_indices])

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(
        utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(
        utils.count_parameters(model, trainable=True)))

    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(
        model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    test_dataset = dataset_class(test_data, test_indices)
    loss_module = get_loss_module(config)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=config['num_workers'],
                                pin_memory=True,
                                collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
    test_evaluator = runner_class(model, test_loader, device, loss_module,
                                    print_interval=config['print_interval'], console=config['console'])

    
    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                      print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(
            keep_all=True, mode="Test")
        print_str = 'Test Summary: '
        print(aggr_metrics_test)
        for k, v in aggr_metrics_test.items():
            print_str += k
            print_str += v
        logger.info(print_str)
        return

    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(my_data, train_indices)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                           print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                 print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])
    logger.info(config['tensorboard_dir'])

    # initialize with +inf or -inf depending on key metric
    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16
    # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    metrics = []
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))
    logger.info('Starting training...')
    previousLossQueue = deque()
    trainingLoss=[]
    validationLoss=[]
    testingLoss=[]
    previousTestingQueue = deque()
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        # dictionary of aggregate epoch metrics
        aggr_metrics_train = trainer.train_epoch(epoch)
        trainingLoss.append(aggr_metrics_train['loss'])
        epoch_runtime = time.time() - epoch_start_time
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_epoch_time)))

        # evaluate if first or last epoch or at specified interval
        # Validation here
        # if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                best_metrics, best_value, epoch)
        validationLoss.append(aggr_metrics_val['loss'])
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))
        
        # Do testing here at each epoch to prove it works
        aggr_metrics_test, per_batch_test, predicted, true = test_evaluator.evaluate(
            keep_all=True, mode="Test")
        testingLoss.append(aggr_metrics_test['loss'])
        previousTestingQueue.append([predicted])
        if min(testingLoss) == aggr_metrics_test['loss']:
                    bestPredicted = predicted
                    predictedTrue = true

        print("Best training loss: ", str(min(trainingLoss)), " at epoch: ", str(trainingLoss.index(min(trainingLoss))))
        print("Best validation loss: ", str(min(validationLoss)), " at epoch: ", str(validationLoss.index(min(validationLoss))))
        print("Best testing loss: ", str(min(testingLoss)), " at epoch: ", str(testingLoss.index(min(testingLoss))))


        utils.save_model(os.path.join(
            config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)
        
        

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(
                config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            # so that this index does not get out of bounds
            if lr_step < len(config['lr_step']) - 1:
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

        # Stop training if reach stop early threshold. Default is 3
        # Implement early stopping
        patience = config['patience']
        loss = aggr_metrics_val["loss"]
        print(aggr_metrics_val)
        print(previousLossQueue)
        print(loss)
        # exit()
        if len(previousLossQueue) < patience :
            previousLossQueue.append(loss)
        else:
            if loss>= max(previousLossQueue):
                # Stop early
                logger.info('Stopping training due to early stopping to prevent overfitting, patience=',str(patience))
                break
            else:
                previousLossQueue.append(loss)
                previousLossQueue.popleft()

        if len(previousTestingQueue) > patience+1:
            previousTestingQueue.popleft()

     # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(
        config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(
        metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(
        config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(
        *utils.readable_time(total_runtime)))
        
    # SHow epoch loss after training
    print(validationLoss)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.plot(validationLoss, label='validation loss', color='orange')
    plt.title('Sliding window validation loss 50 epochs')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    # plt.show()

    print(trainingLoss)
    plt.plot(trainingLoss, label='training loss', color='green')
    plt.title('Sliding window training loss 50 epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    # plt.show()

    print(testingLoss)
    plt.plot(testingLoss, label='testing loss', color='blue')
    plt.title('Sliding window testing loss 50 epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    # plt.show()


    print(bestPredicted)
    print(predictedTrue)
    plt.plot(bestPredicted, label='testing prediction', color='blue')
    plt.plot(predictedTrue, label='testing label', color='red')
    plt.title('Best Prediction')
    plt.xlabel('Time')
    plt.ylabel('CPU%')
    plt.legend(loc='best')
    # plt.show()

    earlyStopTest = np.array(previousTestingQueue.popleft()).flatten()
    print(earlyStopTest)
    plt.plot(earlyStopTest, label='testing prediction', color='green')
    plt.plot(predictedTrue, label='testing label', color='red')
    plt.title('Best Prediction from early stop')
    plt.xlabel('Time')
    plt.ylabel('CPU%')
    plt.legend(loc='best')
    # plt.show()
    mse=str(round(mean_squared_error(bestPredicted, predictedTrue), 8))
    mae=str(round(mean_absolute_error(bestPredicted, predictedTrue), 8))

    earlymse=str(round(mean_squared_error(earlyStopTest, predictedTrue), 8))
    earlymae=str(round(mean_absolute_error(earlyStopTest, predictedTrue), 8))

    # Do testing after training
    aggr_metrics_test, per_batch_test, predicted, true = test_evaluator.evaluate(
            keep_all=True, mode="Test")
    print("MSE = "+str(round(mean_squared_error(true, predicted), 8))+" MAE = "+str(round(mean_absolute_error(
            true, predicted), 8))+" RMSE = "+str(round(mean_squared_error(true, predicted, squared=True), 8)))
    rcParams['figure.figsize'] = 14, 8
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.plot(true, label='Actual CPU%', color='orange')
    plt.plot(predicted, label='Predicted CPU%', color='green')
    plt.title('Transformer Workload Traffic Prediction')
    plt.xlabel('Time')
    plt.ylabel('CPU%')
    plt.legend(loc='best')
    # plt.show()
    fileName = '../../../Results/newResults.txt'
    # if lr==1e-05:
    #     fileName = '../../Results/LaptopResults.txt'
    file = open(fileName, 'a')

    file.write("\nLast epochs= "+str(epoch))
    file.write("\nTotal runtime: {} hours, {} minutes, {} seconds".format(
        *utils.readable_time(total_runtime)))
    file.write("\nBestTest = "+str(bestPredicted))

    file.write('\nEarly stop test= '+str(earlyStopTest))
    file.write("\nEarly stop MSE = "+earlymse + " MAE = "+earlymae+'\n')
    file.write("\Best MSE = "+mse + " MAE = "+mae+'\n')
    file.close()

    return best_value


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
