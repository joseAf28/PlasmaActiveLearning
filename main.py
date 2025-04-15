import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import logging
from pyDOE import lhs
import matplotlib.pyplot as plt
import time 

from sklearn.preprocessing import StandardScaler

import models
import PhysicalModel as env
import ActiveLearning as al


#### setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




if __name__ == '__main__':
    # Configuration for a 2D example
    foldername = 'Figures'
    
    ##* Init Configs
    # config = {
    #     'ensemble_size': 10,
    #     'input_size': 3,         
    #     'hidden_size': 256,
    #     'dropout_rate': 0.1,
    #     'do_dropout': False,
    #     'output_size': 10,
        
    #     'num_epochs': 70,# 150      
    #     'batch_size': 32,
    #     'learning_rate': 1e-2,
        
    #     'candidate_pool_size': 150,
    #     'n_samples': 200,    # 30
    #     'subset_frac': 0.7,
    #     'mc_runs': 1,
    #     'n_actions': 15,
    #     'lambda_vec': np.array([0.2,0.6,0.2])
    # }
    
    ###? new update
    config = {
        'ensemble_size': 10,
        'input_size': 3,         
        'hidden_size': 256,
        'dropout_rate': 0.1,
        'do_dropout': False,
        'output_size': 10,
        
        'num_epochs': 70,
        'batch_size': 32,
        'learning_rate': 1e-2,
        
        'candidate_pool_size': 150,
        'n_samples': 200,    # 30
        'subset_frac': 0.7,
        'mc_runs': 1,
        'n_actions': 15,
        'lambda_vec': np.array([0.15,0.7,0.15])
    }
    
    
    if config['input_size'] == 1:
        bounds = np.array([[0, 2]])
    elif config['input_size'] == 2:
        bounds = np.array([[0, 2], [0, 2]])
    elif config['input_size'] == 3:
        bounds = np.array([[0, 1], [0, 1], [0, 1]])
    else:
        raise ValueError("Input dimension must be 1, 2 or 3")
    
    
    nb_iters = 20  # Number of active learning iterations

    # Generate initial training samples via LHS (no jitter for initial samples)
    x_init = al.SeqModel.generate_lhs_samples_with_jitter(config['n_samples'], config['input_size'], bounds, jitter=0.0)
    
    # Instantiate and initialize the sequential model
    seq_model = al.SeqModel(config, bounds, nb_iters)
    seq_model.init_ensemble(x_init)
    
    loss_test_ensemble_vec = []
    loss_test_baseline_vec = []
    loss_test_baseline_buffer_vec = []
    size_buffer_vec = []
    extra_time_vec = []
    
    time_acc = 0.0
    
    for i in range(nb_iters):
        
        logger.info(f"Iteration {i+1}/{nb_iters}")
        
        time_init = time.time()
        
        state = seq_model.state(i)
        next_samples = seq_model.action(state, n_action=config['n_actions'], lambda_vec=config['lambda_vec'])
        seq_model.train_ensemble(next_samples)
        
        time_end = time.time()
        
        extra_time = time_end - time_init
        time_acc += extra_time
        extra_time_vec.append(time_acc)
        
        ###! generate radom data inside the bouds and check the predictions
        x_input_random = np.random.uniform(bounds[:, 0], bounds[:, 1], (100, config['input_size']))
        x_input_random_tensor = torch.tensor(x_input_random, dtype=torch.float32)
        y_input_random = env.generate_data(x_input_random_tensor)[1].numpy()
        
        ##!# scale the data
        x_input_random_scaled_ensemble = seq_model.standard_scaler_x.transform(x_input_random)
        y_input_random_scaled_ensemble = seq_model.standard_scaler_y.transform(y_input_random)
        x_input_random_tensor_scaled_ensemble = torch.tensor(x_input_random_scaled_ensemble, dtype=torch.float32)
        
        mean_pred_test, sigma_pred_test, _ = al.SeqModel.ensemble_predict(seq_model.ensemble_models, x_input_random_tensor_scaled_ensemble)
        loss = np.mean(np.square(mean_pred_test - y_input_random_scaled_ensemble))
        
        loss_test_ensemble_vec.append(loss)
        
        logger.info(f"State Avg Sigma: {np.mean(state[3])}| Test: MSE Loss: {loss},  Avg Uncertainty: {np.mean(sigma_pred_test)} | Size of buffer: {len(seq_model.buffer[0])}")
        
        
        if config['input_size'] == 1:
            x_test = np.linspace(bounds[0, 0], bounds[0, 1], 500).reshape(-1, 1)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            mu_vec, sigma_vec, _ = al.SeqModel.ensemble_predict(seq_model.ensemble_models, x_test_tensor)
            x_train = seq_model.buffer[0].numpy()
            y_train = seq_model.buffer[1].numpy()
            
            plot_system(x_train, y_train, x_test, mu_vec, sigma_vec, i, next_samples, foldername) 
        
        elif config['input_size'] == 2:
            # Create a grid for plotting in 2D
            num_points_grid = 50
            x1 = np.linspace(bounds[0, 0], bounds[0, 1], num_points_grid)
            x2 = np.linspace(bounds[1, 0], bounds[1, 1], num_points_grid)
            X1, X2 = np.meshgrid(x1, x2)
            x_test = np.column_stack((X1.ravel(), X2.ravel()))
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            
            mu_vec, sigma_vec = al.SeqModel.ensemble_predict(seq_model.ensemble_models, x_test_tensor)
            x_train = seq_model.buffer[0].numpy()
            y_train = seq_model.buffer[1].numpy()
            
            plot_system(x_train, y_train, x_test, mu_vec, sigma_vec, i, next_samples, foldername)
        else:
            pass
            # logger.warning("Plotting not implemented for input dimensions greater than 2.")
        
        
        ###* Baseline Models And Comparisons
        model_standard = models.SimpleNet(input_dim=config['input_size'],
                                hidden_dim=config['hidden_size'],
                                output_dim=config['output_size'],
                                dropout_rate=config['dropout_rate'],
                                do_dropout=config['do_dropout'])
        
        
        model_standard_buffer = models.SimpleNet(input_dim=config['input_size'],
                                hidden_dim=config['hidden_size'],
                                output_dim=config['output_size'],
                                dropout_rate=config['dropout_rate'],
                                do_dropout=config['do_dropout'])
        
        
        config2 = config.copy()
        config2['num_epochs'] = 800
        
        n_samples = len(seq_model.buffer[0])
        
        size_buffer_vec.append(n_samples)
        
        if i == nb_iters - 1:
            time_init = time.time()
        
        standard_scaler_baseline_x = StandardScaler()
        standard_scaler_baseline_y = StandardScaler()
        
        x_train = al.SeqModel.generate_lhs_samples_with_jitter(n_samples, config2['input_size'], bounds, jitter=0.0)
        y_train = env.generate_data(x_train)[1]
        
        x_train_scaled = standard_scaler_baseline_x.fit_transform(x_train)
        y_train_scaled = standard_scaler_baseline_y.fit_transform(y_train)
        x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        
        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
        ### train the model
        model_standard = al.SeqModel.train_network(model_standard, dataset, config2)
        
        dataset_buffer = TensorDataset(seq_model.buffer[0], seq_model.buffer[1])
        model_standard_buffer = al.SeqModel.train_network(model_standard_buffer, dataset_buffer, config2)
        
        if i == nb_iters - 1:
            Delta_time_baseline = time.time() - time_init
        
        print()
        
        ###! predict with the model
        x_input_random_scaled_baseline = standard_scaler_baseline_x.transform(x_input_random)
        y_input_random_scaled_baseline = standard_scaler_baseline_y.transform(y_input_random)
        
        y_test_pred = model_standard(torch.tensor(x_input_random_scaled_baseline, dtype=torch.float32)).detach().numpy()
        loss_baseline = np.mean(np.square(y_test_pred - y_input_random_scaled_baseline))
        logger.info(f"Baseline Model Test: MSE Loss: {loss_baseline} at iteration {i}")
        
        loss_test_baseline_vec.append(loss_baseline)
        
        y_test_pred_buffer = model_standard_buffer(x_input_random_tensor_scaled_ensemble).detach().numpy()
        loss_baseline_buffer = np.mean(np.square(y_test_pred_buffer - y_input_random_scaled_ensemble))
        logger.info(f"Baseline Buffer Model Test: MSE Loss: {loss_baseline_buffer} at iteration {i}")
        
        loss_test_baseline_buffer_vec.append(loss_baseline_buffer)
        
        print("------------------------------------------------")
    
    
    ### save ensemble models and the buffer 
    torch.save(seq_model.ensemble_models, 'ensemble_models.pth')
    torch.save(seq_model.buffer, 'buffer.pth')
    
    ### save the scaler
    torch.save(seq_model.standard_scaler_x, 'standard_scaler_x.pth')
    torch.save(seq_model.standard_scaler_y, 'standard_scaler_y.pth')
    
    
    ### plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(size_buffer_vec, loss_test_baseline_vec, label='Baseline Model', marker='x')
    plt.plot(size_buffer_vec, loss_test_ensemble_vec, label='Ensemble Model', marker='o')
    plt.plot(size_buffer_vec, loss_test_baseline_buffer_vec, label='Baseline Buffer Model', marker='s')
    plt.xlabel('Buffer Size')
    plt.ylabel('MSE Loss')
    plt.title(f'Ensemble Gain MSE: {round(100*np.abs(loss_test_baseline_vec[-1] - loss_test_ensemble_vec[-1])/loss_test_baseline_vec[-1], 3)} % \
            | Buffer Gain: {round(100*np.abs(loss_test_baseline_vec[-1] - loss_test_baseline_buffer_vec[-1])/loss_test_baseline_vec[-1], 3)} %')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig(f"{foldername}/loss_curves_{config["ensemble_size"]}_{config['lambda_vec']}_{nb_iters}_{config['ensemble_size']}.png")
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(size_buffer_vec, extra_time_vec, label='Time per Iteration', marker='o')
    plt.scatter(size_buffer_vec[-1], Delta_time_baseline, color='red', label='Baseline Time')
    plt.xlabel('Buffer Size')
    plt.ylabel('Time (s)')
    plt.title(f"ratio: {round(extra_time_vec[-1] / Delta_time_baseline, 3)}")
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig(f"{foldername}/time_curves_{config["ensemble_size"]}_{config['lambda_vec']}_{nb_iters}_{config['ensemble_size']}.png")
    