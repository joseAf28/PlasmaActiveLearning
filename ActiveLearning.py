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


import models
import PhysicalModel as env



###* Sequential Model for Active Learning
###! For now, I am considering a version of the BALD score as the acquisition function
###? Later try to implement the EPIG score function

class SeqModel:

    def __init__(self, config: dict, bounds: np.ndarray, T: int):

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
        
        self.buffer = None          # Aggregated training data as tuple (x, y)
        self.ensemble_mu = []       # History of ensemble mean predictions
        self.ensemble_sigma = []    # History of ensemble stds
    
    
    ###* Methods to sample the candidate pool - the state
    @staticmethod
    def generate_lhs_samples_with_jitter(n_samples: int, input_dim: int, 
                                        bounds: np.ndarray, jitter: float = 0.0) -> torch.Tensor:
        
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
    def jitter_schedule(t: int, T: int, sigma_init: float = 0.001, sigma_max: float = 0.15) -> float:
        
        sigma_t = sigma_init + (sigma_max - sigma_init) * (t / T)
        return sigma_t
    
    
    ###* Acquisition function
    @staticmethod
    def acquisition_score(candidate: np.ndarray, uncertainty: float, S: np.ndarray, lambda_diversity: float, tolerance: float = 1e-2) -> float:
        
        if S.size == 0:
            diversity = 0.0
        else:
            distances = np.linalg.norm(S - candidate, axis=1)
            diversity = np.min(distances)
        return uncertainty + lambda_diversity * diversity
    
    
    ###* Greedy selection of candidates - the action
    @staticmethod
    def greedy_selection(candidates: torch.Tensor, uncertainties: np.ndarray, buffer: np.ndarray, n_action: int, lambda_diversity: float = 1.0, tolerance: float = 1e-3) -> np.ndarray:
        
        candidates_np = candidates.numpy()
        if buffer is None or len(buffer) == 0:
            S = np.empty((0, candidates_np.shape[1]))
        else:
            S = buffer.copy()
        
        # Compute minimum distances (vectorized)
        if S.size == 0:
            min_distances = np.full(len(candidates_np), np.inf)
        else:
            # distances = np.linalg.norm(candidates_np[:, None, :] - S[None, :, :], axis=2)
            distances = np.sum(np.abs(candidates_np[:, None, :] -  S[None, :, :]), axis=2)
            min_distances = distances.min(axis=1)
        
        valid_indices = np.where(min_distances > tolerance)[0]
        if valid_indices.size == 0:
            raise ValueError('No candidates are far enough from the buffer')
        
        scores = np.array([SeqModel.acquisition_score(candidates_np[i], uncertainties[i], S, lambda_diversity, tolerance) for i in valid_indices])
        
        top_indices = valid_indices[np.argpartition(-scores, n_action)[:n_action]]
        return candidates_np[top_indices]
    
    
    @staticmethod
    def train_network(model: nn.Module, dataset: TensorDataset, config: dict) -> nn.Module:
        
        subset_size = int(len(dataset) * config['subset_frac'])
        indices = np.random.choice(len(dataset), size=subset_size, replace=True)
        bootstrap_dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(bootstrap_dataset, batch_size=config['batch_size'], shuffle=True)
        
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
    
    
    @staticmethod
    def ensemble_predict(ensemble: list, x_input: torch.Tensor, mc_dropout: bool = False, mc_runs: int = 10, entropy_flag=False) -> (np.ndarray, np.ndarray):
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
            
            score = np.mean(std_pred, axis=-1)                             # (n_samples,)
            
            return mean_pred, std_pred, score
        
        
    def train_ensemble(self, x_new: np.ndarray, memory_replay_fraction: float = 0.3) -> None:
        
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        x_new_tensor, y_new_tensor = env.generate_data(x_new_tensor)
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
            self.ensemble_models[i] = SeqModel.train_network(model, combined_dataset, self.config)
        
        
        
    def update_buffer(self, x_new: torch.Tensor, y_new: torch.Tensor) -> None:
        
        if self.buffer is None:
            self.buffer = (x_new, y_new)
        else:
            x_old, y_old = self.buffer
            self.buffer = (torch.cat((x_old, x_new), dim=0), torch.cat((y_old, y_new), dim=0))
    
    
    ###* MDP functions
    
    def init_ensemble(self, x_init: torch.Tensor) -> None:
        ###* Initialize the ensemble with a set of initial points
        
        x_init, y_init = env.generate_data(x_init)
        dataset = TensorDataset(x_init, y_init)
        for i, model in enumerate(self.ensemble_models):
            self.ensemble_models[i] = SeqModel.train_network(model, dataset, self.config)
        self.update_buffer(x_init, y_init)
    
    
    def state(self, t: int) -> torch.Tensor:
        
        sigma_t = SeqModel.jitter_schedule(t, self.T)
        x_candidates = SeqModel.generate_lhs_samples_with_jitter(self.config['candidate_pool_size'], self.config['input_size'], self.bounds, jitter=sigma_t)
        mean_pred, std_pred, score = SeqModel.ensemble_predict(self.ensemble_models, x_candidates)
        
        
        self.ensemble_mu.append(mean_pred)
        self.ensemble_sigma.append(std_pred)
        return (x_candidates, mean_pred, std_pred, score)
    
    
    def action(self, state, n_action: int = 20, lambda_diversity: float = 0.2) -> np.ndarray:

        x_candidates, mu_vec, sigma_vec, score = state
        
        buffer_points = self.buffer[0].numpy() if self.buffer is not None else None
        next_samples = SeqModel.greedy_selection(x_candidates, score, buffer_points, n_action, lambda_diversity)
        return next_samples