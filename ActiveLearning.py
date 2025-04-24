import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from pyDOE import lhs
import matplotlib.pyplot as plt
import time 

import models


###* Sequential Model for Active Learning
class SeqModel:

    def __init__(self, config, bounds, T, simulator):
        
        self.config = config
        if not (isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] == 2):
            raise ValueError("bounds must be a numpy array of shape (input_dim, 2)")
        self.bounds = bounds
        self.T = T
        self.input_dim = config['input_size']
        self.ensemble_models = []
        for i in range(config['ensemble_size']):
            model = models.SimpleNet(input_dim=config['input_size'],
                            hidden_dim=config['hidden_size'],
                            output_dim=config['output_size'],
                            index=i,
                            dropout_rate=config['dropout_rate'],
                            do_dropout=config['do_dropout'])
            self.ensemble_models.append(model)
        
        self.simulator = simulator
        
        self.buffer = None          # Aggregated training data as tuple (x, y)
        self.ensemble_mu = []       # History of ensemble mean predictions
        self.ensemble_sigma = []    # History of ensemble stds
        
        self.standard_scaler_x = StandardScaler()
        self.standard_scaler_y = StandardScaler()
    
    
    ###* Methods to sample the candidate pool - the state
    @staticmethod
    def generate_lhs_samples_with_jitter(n_samples: int, input_dim: int, bounds: np.ndarray, jitter: float = 0.0):
        if bounds.shape[0] != input_dim:
            raise ValueError("Bounds array must have shape (input_dim, 2)")
            
        raw_samples = lhs(input_dim, samples=n_samples)  # (n_samples, input_dim)
        scaled_samples = bounds[:, 0] + raw_samples * (bounds[:, 1] - bounds[:, 0])
        
        if jitter > 0.0:
            noise = np.random.normal(0, jitter, size=scaled_samples.shape)
            scaled_samples += noise
            scaled_samples = np.clip(scaled_samples, bounds[:, 0], bounds[:, 1])
        
        return torch.tensor(scaled_samples, dtype=torch.float32)
    
    
    @staticmethod
    def jitter_schedule(t: int, T: int, sigma_init: float = 0.001, sigma_max: float = 0.15):
        
        sigma_t = sigma_init + (sigma_max - sigma_init) * (t / T)
        return sigma_t
    
    
    @staticmethod
    def train_network(model: nn.Module, dataset: TensorDataset, config: dict, flag_init=False):
        
        if flag_init:
            dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        else:
            subset_size = int(len(dataset) * config['subset_frac'])
            indices = np.random.choice(len(dataset), size=subset_size, replace=True)
            bootstrap_dataset = torch.utils.data.Subset(dataset, indices)
            dataloader = DataLoader(bootstrap_dataset, batch_size=config['batch_size'], shuffle=True)
            
        if flag_init:
            num_epochs = config['num_epoch_init']
        else:
            num_epochs = config['num_epochs']
        
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()
        model.train()
        
        for epoch in range(num_epochs):
            for batch_x, batch_y in dataloader:
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        return model
    
    
    @staticmethod
    def ensemble_predict(ensemble: list, x_input: torch.Tensor, mc_dropout: bool = False, mc_runs: int = 10, entropy_flag=False):
        predictions = []
        for model in ensemble:
            if mc_dropout:
                model.train()
            else:
                model.eval()
            
            model_preds = []
            runs = mc_runs if mc_dropout else 1
            with torch.no_grad():
                for _ in range(runs):
                    pred = model(x_input).numpy()
                    model_preds.append(pred)
        
            model_preds = np.mean(np.array(model_preds), axis=0)
            predictions.append(model_preds)
        
        ### predictions : (ensemble_size, n_samples, output_dim)
        predictions = np.array(predictions).transpose((1, 0, 2))    # (n_samples, ensemble_size, output_dim)
        
        
        if entropy_flag:
            #### log det Covariance way
            mean_pred = np.mean(predictions, axis=1, keepdims=True)     # (n_samples, output_dim)
            centered_pred = predictions - mean_pred                     # (n_samples, ensemble_size, output_dim)
            
            centered_pred_transposed = centered_pred.transpose((0, 2, 1))  # (n_samples, output_dim, ensemble_size)
            covariance_pred = np.matmul(centered_pred_transposed, centered_pred) / (len(ensemble) - 1) # (n_samples, output_dim, output_dim)
            
            det_pred = np.linalg.det(covariance_pred)
            sign, logdet = np.linalg.slogdet(covariance_pred)
            
            entropy_pred = 0.5 * logdet
            
            mean_pred = mean_pred.squeeze()
            std_pred = np.sqrt(np.diagonal(covariance_pred, axis1=1, axis2=2))  # (n_samples, output_dim)
            
            return mean_pred, std_pred, entropy_pred
        else:
            mean_pred = np.mean(predictions, axis=1)                     # (n_samples, output_dim)
            std_pred = np.std(predictions, axis=1)                       # (n_samples, output_dim)
            
            score = np.mean(std_pred, axis=-1)                           # (n_samples,)
            
            return mean_pred, std_pred, score
    
    
    def train_ensemble(self, x_new: np.ndarray, memory_replay_fraction: float = 0.3):
        
        ###* run the simulator to get the new data
        y_new, _ = self.simulator.run_simulation(x_new)
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        y_new_tensor = torch.tensor(y_new, dtype=torch.float32)
        
        ###! Scale the new data using the old scaler
        x_new_scaled = self.standard_scaler_x.transform(x_new_tensor.numpy())
        y_new_scaled = self.standard_scaler_y.transform(y_new_tensor.numpy())
        x_new_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)
        y_new_tensor = torch.tensor(y_new_scaled, dtype=torch.float32)
        
        new_dataset = TensorDataset(x_new_tensor, y_new_tensor)
        
        if self.buffer is not None:
            x_old, y_old = self.buffer
            n_old = int(memory_replay_fraction * len(x_old))
            indices = np.random.choice(len(x_old), size=n_old, replace=False)
            x_old_subset = x_old[indices]
            y_old_subset = y_old[indices]
            old_dataset = TensorDataset(x_old_subset, y_old_subset)
            combined_dataset = ConcatDataset([new_dataset, old_dataset])
        else:
            combined_dataset = new_dataset
        
        self.update_buffer(x_new_tensor, y_new_tensor)
        
        for i, model in enumerate(self.ensemble_models):
            self.ensemble_models[i] = SeqModel.train_network(model, combined_dataset, self.config, flag_init=False)
        
        
        
    def update_buffer(self, x_new: torch.Tensor, y_new: torch.Tensor):
        
        if self.buffer is None:
            self.buffer = (x_new, y_new)
        else:
            x_old, y_old = self.buffer
            self.buffer = (torch.cat((x_old, x_new), dim=0), torch.cat((y_old, y_new), dim=0))
    
    
    ###* MDP functions
    def init_ensemble(self, x_init: torch.Tensor, y_init: torch.Tensor = None, pbar=None):
        ###* Initialize the ensemble with a set of initial points
        
        if y_init is None:
            y_init, _ = self.simulator.run_simulation(x_init.numpy(), pbar=pbar)
            y_init = torch.tensor(y_init, dtype=torch.float32)
        
        ###! Scale the initial data
        x_init_scaled = self.standard_scaler_x.fit_transform(x_init.numpy())
        y_init_scaled = self.standard_scaler_y.fit_transform(y_init.numpy())
        x_init = torch.tensor(x_init_scaled, dtype=torch.float32)
        y_init = torch.tensor(y_init_scaled, dtype=torch.float32)
        
        dataset = TensorDataset(x_init, y_init)
        for i, model in enumerate(self.ensemble_models):
            self.ensemble_models[i] = SeqModel.train_network(model, dataset, self.config, flag_init=True)
        self.update_buffer(x_init, y_init)
    
    
    def compute_expected_gradients_X(self, x_input, mean_pred, sigma_pred):
        ###! taking the gradients of the loss function w.r.t. the input
        
        mean_pred_tensor = torch.tensor(mean_pred, dtype=torch.float32)
        x_input_tensor = torch.tensor(x_input, dtype=torch.float32, requires_grad=True)
        
        expected_gradients = []
        for model in self.ensemble_models:
            
            model.eval()
            model_pred_tensor = model(x_input_tensor)
            y_sample_tensor = torch.tensor(np.random.normal(mean_pred, sigma_pred), dtype=torch.float32)
            
            loss = nn.MSELoss()(model_pred_tensor, y_sample_tensor)
            
            if x_input_tensor.grad is not None:
                x_input_tensor.grad.zero_()
            
            gradients = torch.autograd.grad(loss, x_input_tensor, create_graph=True)[0]
            
            grad_norm = torch.norm(gradients, dim=1)
            expected_gradients.append(grad_norm.detach().numpy())
        
        expected_gradients = np.array(expected_gradients).transpose((1, 0))     # (n_samples, ensemble_size)
        expected_gradients = np.mean(expected_gradients, axis=1)                # (n_samples,)
        
        return expected_gradients
    
    
    def compute_expected_gradients(self, x_input, mean_pred, sigma_pred):
        ###! taking the gradients of the loss function w.r.t. the model parameters
        ###* include a vectorized version of this function later
        
        gradients_norms_ensemble = []
        for i in range(x_input.shape[0]):
            
            x = torch.tensor(x_input[i:i+1], dtype=torch.float32, requires_grad=True)
            
            gradients_norms = []
            for model in self.ensemble_models:
                model.eval()
                model_pred = model(x)
                
                y_sample = np.random.normal(mean_pred[i], sigma_pred[i])
                y_sample_tensor = torch.tensor(y_sample, dtype=torch.float32).unsqueeze(0)
                
                loss = nn.MSELoss()(model_pred, y_sample_tensor)
                
                model.zero_grad()
                loss.backward()
                
                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        total_norm += torch.norm(param.grad).item() ** 2
                
                total_norm_sq = total_norm ** 0.5
                gradients_norms.append(total_norm_sq)
                
            gradients_norms = np.array(gradients_norms)
            gradients_norms_ensemble.append(gradients_norms.mean())
    
        return gradients_norms_ensemble
    
    
    def compute_distances(self, candidates: np.ndarray, buffer: np.ndarray):
        if buffer is None or len(buffer) == 0:
            return np.full(len(candidates), 1e3)
        else:
            distances = np.linalg.norm(candidates[:, None, :] -  buffer[None, :, :], axis=2, ord=1)
            return distances.min(axis=1)
    
    
    def state(self, t: int):
        
        ###* candidate pool scaled
        sigma_t = SeqModel.jitter_schedule(t, self.T)
        x_candidates = SeqModel.generate_lhs_samples_with_jitter(self.config['candidate_pool_size'], self.config['input_size'], self.bounds, jitter=sigma_t)
        x_candidates_scaled = torch.tensor(self.standard_scaler_x.transform(x_candidates.numpy()), dtype=torch.float32)
        
        mean_pred, std_pred, uncertainty = SeqModel.ensemble_predict(self.ensemble_models, x_candidates_scaled)
        
        expected_gradient = self.compute_expected_gradients(x_candidates_scaled.numpy(), mean_pred, std_pred)
        distances = self.compute_distances(x_candidates_scaled.numpy(), self.buffer[0].numpy() if self.buffer is not None else None)
        
        self.ensemble_mu.append(mean_pred)
        self.ensemble_sigma.append(std_pred)
        return (x_candidates_scaled, mean_pred, std_pred, uncertainty, expected_gradient, np.reciprocal(distances))
    
    
    @staticmethod
    def acquisition_function(uncertainty_vec, expected_gradient_vec, distances_vec, lambda_vec):
        
        ### normalize the vectors
        uncertainty_vec = (uncertainty_vec - np.min(uncertainty_vec)) / (np.max(uncertainty_vec) - np.min(uncertainty_vec))
        expected_gradient_vec = (expected_gradient_vec - np.min(expected_gradient_vec)) / (np.max(expected_gradient_vec) - np.min(expected_gradient_vec))
        distances_vec = (distances_vec - np.min(distances_vec)) / (np.max(distances_vec) - np.min(distances_vec))
        
        lambda_uncertainty = lambda_vec[0]
        lambda_expected_gradient = lambda_vec[1]
        lambda_diversity = lambda_vec[2]
        
        score = lambda_uncertainty * uncertainty_vec + lambda_expected_gradient * expected_gradient_vec + lambda_diversity * distances_vec
        return score
    
    
    def action(self, state, n_action: int = 20, lambda_vec=[0.4, 0.4, 0.2]):
        x_candidates_scaled, mu_vec, sigma_vec, uncertainty, expected_gradient, distances = state
        
        score_vec = SeqModel.acquisition_function(uncertainty, expected_gradient, distances, lambda_vec)
        
        #### greedy selection of candidates: top-n_action elements
        state_top_k = np.argpartition(-score_vec, n_action)[:n_action]
        
        next_samples = x_candidates_scaled.numpy()[state_top_k]
        next_samples_unscaled = self.standard_scaler_x.inverse_transform(next_samples)
        
        return next_samples_unscaled