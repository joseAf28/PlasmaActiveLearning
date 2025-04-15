import numpy as np
import torch 


####* Physical Model for now
def generate_data(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    x_np = x.numpy()
    n_samples = x_np.shape[0]
    
    if x_np.shape[1] == 1:
        y = np.sin(np.pi * x_np)
    elif x_np.shape[1] == 2:
        y = np.sin(np.pi * x_np[:, 0:1]) * np.cos(np.pi * x_np[:, 1:2])
    else:
        y1 = np.sin(np.pi * x_np[:, 0:1])             
        y2 = np.cos(np.pi * x_np[:, 1:2])             
        y3 = x_np[:, 0:1]**2                          
        y4 = x_np[:, 1:2]**3                          
        y5 = np.tanh(x_np[:, 2:3])                  
        
        y6 = y1 + y2                                
        y7 = y3 - y2                                
        y8 = y1 * y4                                
        y9 = np.sin(x_np[:, 2:3])                    
        y10 = np.exp(-np.abs(x_np[:, 0:1]))        
        
        y = np.concatenate(np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]), axis=1)
        
    return x, torch.tensor(y, dtype=torch.float32)

