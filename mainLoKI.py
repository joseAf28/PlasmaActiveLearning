import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import logging
from pyDOE import lhs
import matplotlib.pyplot as plt
import time 
import matlab.engine
import h5py
from tqdm import tqdm

import models
import LokiCaller as env
import ActiveLearning as al


#### setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    
    ###* setup the Physical Simulator 
    logger.info("Setting up the Physical Simulator")
    
    file_path_sim = 'LoKI_v3/Code'
    x_names = ['wallTemperature', 'gasPressure', 'electronDensity']
    bounds = np.array([[300, 400], [300, 900], [1e15, 1e16]])
    
    eng = matlab.engine.start_matlab()
    eng.cd(file_path_sim, nargout=0)
    
    simulator = env.PhysicalSimulator(eng, file_path_sim, 'setup_O2_simple_mod.in', 'OxygenSimplified_1/chemFinalDensities.txt', x_names)
    
    ###* test sets
    with h5py.File('test_set.hdf5', 'r') as f:
        x_input_test = f['x_samples'][:]
        y_input_test = f['y_samples'][:]
    
    
    input_dim = len(x_names)
    output_dim = y_input_test.shape[1]
    
    ###* active learning task
    logger.info("Setting up the Active Learning Task")
    
    foldername = 'FiguresNew'
    
    config = {
        'ensemble_size': 10,
        'input_size': input_dim,         
        'hidden_size': 256,
        'dropout_rate': 0.1,
        'do_dropout': False,
        'output_size': output_dim,
        
        'num_epoch_init': 400,
        'num_epochs': 70,
        'batch_size': 32,
        'learning_rate': 1e-2,
        
        'candidate_pool_size': 150,
        'n_samples': 150,    # 200
        'subset_frac': 0.7,
        'mc_runs': 1,
        'n_actions': 15,
        'lambda_vec': np.array([0.12,0.85,0.03])
    }
    
    nb_iters = 23  # Number of active learning iterations

    ##* use instead the data for the initial buffer
    ## Generate initial training samples via LHS (no jitter for initial samples)
    # x_init = al.SeqModel.generate_lhs_samples_with_jitter(config['n_samples'], config['input_size'], bounds, jitter=0.0)
    with h5py.File('inital_buffer.hdf5', 'r') as f:
        x_init = f['x_samples'][:]
        y_init = f['y_samples'][:]
    
    
    loss_test_ensemble_vec = []
    iters_vec = []
    buffer_size_vec = []
    active_time_vec = []
    
    # Instantiate and initialize the sequential model
    seq_model = al.SeqModel(config, bounds, nb_iters, simulator)
    
    time_init = time.time()
    
    x_init_tensor = torch.tensor(x_init, dtype=torch.float32)
    y_init_tensor = torch.tensor(y_init, dtype=torch.float32)
    seq_model.init_ensemble(x_init_tensor, y_init=y_init_tensor)
    
    time_init_buffer = time.time() - time_init
    
    ##* init prediction
    x_input_test_scaled = seq_model.standard_scaler_x.transform(x_input_test)
    y_input_test_scaled = seq_model.standard_scaler_y.transform(y_input_test)
    x_input_test_scaled_tensor = torch.tensor(x_input_test_scaled, dtype=torch.float32)
    
    mean_pred_test, sigma_pred_test, _ = al.SeqModel.ensemble_predict(seq_model.ensemble_models, x_input_test_scaled_tensor)
    loss = np.mean(np.square(mean_pred_test - y_input_test_scaled))
    logger.info(f"Initial Test: MSE Loss: {loss},  Avg Uncertainty: {np.mean(sigma_pred_test)} | Size of buffer: {len(seq_model.buffer[0])}")
    
    loss_test_ensemble_vec.append(loss)
    buffer_size_vec.append(len(seq_model.buffer[0]))
    active_time_vec.append(time_init_buffer)
    
    time_acc = time_init_buffer
    for i in range(nb_iters):
        
        logger.info(f"Iteration {i+1}/{nb_iters}")
        
        time_init = time.time()
        
        state = seq_model.state(i)
        logger.debug(f"State: {state}")
        
        next_samples = seq_model.action(state, n_action=config['n_actions'], lambda_vec=config['lambda_vec'])
        logger.debug(f"Next samples: {next_samples}")
        
        seq_model.train_ensemble(next_samples)
        logger.debug(f"Trained ensemble with samples: {next_samples}")
        
        delta_time =time.time() - time_init
        time_acc += delta_time
        active_time_vec.append(time_acc)
        
        
        ###* test the ensemble
        mean_pred_test, sigma_pred_test, _ = al.SeqModel.ensemble_predict(seq_model.ensemble_models, x_input_test_scaled_tensor)
        loss = np.mean(np.square(mean_pred_test - y_input_test_scaled))
        
        loss_test_ensemble_vec.append(loss)
        buffer_size_vec.append(len(seq_model.buffer[0]))
        iters_vec.append(i+1)
        
        logger.info(f"State Avg Sigma: {np.mean(state[3])}| Test: MSE Loss: {loss},  Avg Uncertainty: {np.mean(sigma_pred_test)} | Size of buffer: {len(seq_model.buffer[0])}")
    
    
    ###* saved data and metrics in hdf5 file
    with h5py.File('active_learning_results.hdf5', 'w') as f:
        f.create_dataset('loss_test_ensemble_vec', data=loss_test_ensemble_vec)
        f.create_dataset('buffer_size_vec', data=buffer_size_vec)
        f.create_dataset('iters_vec', data=iters_vec)
        f.create_dataset('active_time_vec', data=active_time_vec)
        f.create_dataset('buffer', data=seq_model.buffer[0])
    
    
    ### save ensemble models and the buffer 
    torch.save(seq_model.ensemble_models, 'saved_models/ensemble_models.pth')
    torch.save(seq_model.standard_scaler_x, 'saved_models/standard_scaler_x.pth')
    torch.save(seq_model.standard_scaler_y, 'saved_models/standard_scaler_y.pth')