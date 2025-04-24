import numpy as np
import matplotlib.pyplot as plt
import time 
import h5py
import logging
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy

import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_network(model: nn.Module, dataset: TensorDataset, config: dict):
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(config['num_epochs']):
        for batch_x, batch_y in dataloader:
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    return model


if __name__ == '__main__':
    
    
    ###* test sets
    with h5py.File('test_set.hdf5', 'r') as f:
        x_input_test = f['x_samples'][:]
        y_input_test = f['y_samples'][:]
    
    
    ###* Buffer size
    n_samples_vec = np.array([150, 200, 250, 300, 350, 400, 450, 500]) ###! fix this later
    x_input_baseline_buffer = []
    y_input_baseline_buffer = []
    with h5py.File('baseline_buffer.hdf5', 'r') as f:
        for i in range(len(n_samples_vec)):
            x_input_baseline_buffer.append(f[f'x_samples_{i}'][:])
            y_input_baseline_buffer.append(f[f'y_samples_{i}'][:])
    
    ####* loss_vec files
    with h5py.File('active_learning_results.hdf5', 'r') as f:
        buffer_size_vec = f['buffer_size_vec'][:]
        loss_test_ensemble_vec = f['loss_test_ensemble_vec'][:]
        buffer_active_learning = f['buffer'][:]
        active_time_buffer_vec = f['active_time_vec'][:]
    
    
    input_dim = x_input_test.shape[1]
    output_dim = y_input_test.shape[1]
    
    config = {
        'ensemble_size': 10,
        'input_size': input_dim,         
        'hidden_size': 256,
        'dropout_rate': 0.1,
        'do_dropout': False,
        'output_size': output_dim,
        
        'num_epochs': 600, # 600
        'batch_size': 32,
        'learning_rate': 1e-2,
    }
    
    loss_test_baseline_I_vec = []
    baseline_time_buffer_vec = []
    
    for j in range(len(n_samples_vec)):
        
        time_init = time.time()
        ensemble_model = []
        for i in range(config['ensemble_size']):
            model = models.SimpleNet(input_dim=config['input_size'],
                            hidden_dim=config['hidden_size'],
                            output_dim=config['output_size'],
                            index=i,
                            dropout_rate=config['dropout_rate'],
                            do_dropout=config['do_dropout'])
            ensemble_model.append(model)
        
        standard_scaler_x = StandardScaler()
        standard_scaler_y = StandardScaler()
        
        ###* train the model
        logger.info("Training the model")
        
        x_input_scaled = standard_scaler_x.fit_transform(x_input_baseline_buffer[j])
        y_input_scaled = standard_scaler_y.fit_transform(y_input_baseline_buffer[j])
        
        x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)
        y_input_tensor = torch.tensor(y_input_scaled, dtype=torch.float32)
        
        dataset = TensorDataset(x_input_tensor, y_input_tensor)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        for model in ensemble_model:
            model = train_network(model, dataset, config)
        
        time_buffer = time.time() - time_init
        baseline_time_buffer_vec.append(time_buffer)
        
        ##* prediction
        x_input_test_scaled = standard_scaler_x.transform(x_input_test)
        x_input_test_tensor = torch.tensor(x_input_test_scaled, dtype=torch.float32)
        y_input_test_scaled = standard_scaler_y.transform(y_input_test)
        y_input_test_tensor = torch.tensor(y_input_test_scaled, dtype=torch.float32)
        
        loss_value = 0.0
        for model in ensemble_model:
            model.eval()
            with torch.no_grad():
                y_pred = model(x_input_test_tensor)
                
                loss = nn.MSELoss()(y_pred, y_input_test_tensor)
                loss_value += loss.item()
            
        loss_value /= config['ensemble_size']
        loss_test_baseline_I_vec.append(loss_value)
        
        logger.info(f"Iter: {j+1}/{len(n_samples_vec)} Test loss: {loss_value}")


    ###* plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(buffer_size_vec, np.sqrt(loss_test_ensemble_vec), label='Test Loss Ensemble', marker='o')
    plt.plot(n_samples_vec, np.sqrt(loss_test_baseline_I_vec), label='Test Loss Baseline', marker='x')
    plt.xlabel('Buffer Size')
    plt.ylabel('RMSE Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.savefig('test_loss_comparison_LoKI.png')
    
    
    loss_baseline_interp = scipy.interpolate.interp1d(n_samples_vec, loss_test_baseline_I_vec, kind='linear', fill_value='extrapolate')
    loss_active_interp = scipy.interpolate.interp1d(buffer_size_vec, loss_test_ensemble_vec, kind='linear', fill_value='extrapolate')
    
    lin_vec = np.linspace(buffer_size_vec[0], buffer_size_vec[-1], 100)
    loss_test_baseline_lin = loss_baseline_interp(lin_vec)
    loss_test_ensemble_lin = loss_active_interp(lin_vec)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lin_vec, (np.sqrt(loss_test_baseline_lin)-np.sqrt(loss_test_ensemble_lin))/np.sqrt(loss_test_baseline_lin)*100, label='% Gain')
    plt.xlabel('Buffer Size')
    plt.ylabel('% Gain RMSE')
    plt.title('Gain in RMSE')
    plt.grid()
    plt.savefig('gain_rmse_LoKI.png')
    
    baseline_time_buffer_vec = np.array(baseline_time_buffer_vec)
    active_time_buffer_vec = np.array(active_time_buffer_vec)
    
    baseline_time_buffer_mod_vec = baseline_time_buffer_vec + 1.5 * n_samples_vec
    active_time_buffer_mod_vec = active_time_buffer_vec + 1.5 * n_samples_vec[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(buffer_size_vec, active_time_buffer_mod_vec, label='Active Time Buffer', marker='o')
    plt.plot(n_samples_vec, baseline_time_buffer_mod_vec, label='Baseline Time Buffer', marker='x')
    plt.xlabel('Buffer Size')
    plt.ylabel('Time (s)')
    plt.title('Time Comparison')
    plt.legend()
    plt.grid()
    plt.savefig('time_comparison_LoKI.png')
    
    
    interp_active = scipy.interpolate.interp1d(buffer_size_vec, active_time_buffer_mod_vec, kind='linear', fill_value='extrapolate')
    interp_baseline = scipy.interpolate.interp1d(n_samples_vec, baseline_time_buffer_mod_vec, kind='linear', fill_value='extrapolate')
    
    lin_vec = np.linspace(buffer_size_vec[0], buffer_size_vec[-1], 100)
    active_time_buffer_mod_vec = interp_active(lin_vec)
    baseline_time_buffer_mod_vec = interp_baseline(lin_vec)
    ratio_vec = active_time_buffer_mod_vec / baseline_time_buffer_mod_vec
    
    plt.figure(figsize=(10, 6))
    plt.plot(lin_vec, ratio_vec, label='Active Time / Baseline Time')
    plt.xlabel('Buffer Size')
    plt.ylabel('Ratio')
    plt.title('Active Time vs Baseline Time')
    plt.legend()
    plt.grid()
    plt.savefig('active_baseline_time_ratio_LoKI.png')
    