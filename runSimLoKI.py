import matlab.engine
import numpy as np
import h5py
from pyDOE import lhs
from tqdm import tqdm

import LokiCaller as env


if __name__ == '__main__':
    
    ##* setup the input parameters
    input_dim = 3
    x_names = ['wallTemperature', 'gasPressure', 'electronDensity']
    bounds = np.array([[300, 400], [300, 900], [1e15, 1e16]])
    jitter = 0.2
    
    file_path = 'LoKI_v3/Code'
    eng = matlab.engine.start_matlab()
    eng.cd(file_path, nargout=0)
    
    system = env.PhysicalSimulator(eng, file_path, 'setup_O2_simple_mod.in', 'OxygenSimplified_1/chemFinalDensities.txt', x_names)
    ####* baseline buffer size
    n_samples_vec = np.array([150, 200, 250, 300, 350, 400, 450, 500])
    
    pbar = tqdm(total=np.sum(n_samples_vec), desc="Running simulations", unit="sample")
    vector_x_samples = []
    vector_y_samples = []
    
    for n_samples in n_samples_vec:
        
        raw_samples = lhs(input_dim, samples=n_samples)
        scaled_samples = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * raw_samples
        y_samples, y_names = system.run_simulation(scaled_samples, pbar=pbar)
        
        vector_x_samples.append(scaled_samples)
        vector_y_samples.append(y_samples)
        
    pbar.close()
    
    with h5py.File('baseline_buffer.hdf5', 'w') as f:
        for i, n_samples in enumerate(n_samples_vec):
            f.create_dataset(f'x_samples_{i}', data=vector_x_samples[i])
            f.create_dataset(f'y_samples_{i}', data=vector_y_samples[i])
            f.create_dataset(f'n_samples_vec_{i}', data=n_samples)
    
    
    ###* test set
    n_samples = 80
    raw_samples = lhs(input_dim, samples=n_samples)
    scaled_samples = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * raw_samples
    noise = np.random.normal(0, 0.1, scaled_samples.shape)
    scaled_samples += noise
    x_samples = np.clip(scaled_samples, bounds[:, 0], bounds[:, 1])
    
    pbar = tqdm(total=n_samples, desc="Running simulations Test Set", unit="sample")
    y_samples, y_names = system.run_simulation(x_samples, pbar=pbar)
    pbar.close()
    
    
    with h5py.File('test_set.hdf5', 'w') as f:
        f.create_dataset('x_samples', data=x_samples)
        f.create_dataset('x_names', data=np.array(x_names, dtype='S'))
        
        f.create_dataset('y_samples', data=y_samples)
        f.create_dataset('y_names', data=np.array(y_names, dtype='S'))
        
    
    
    ####* initial buffer size
    n_samples = 150
    raw_samples = lhs(input_dim, samples=n_samples)
    x_samples_buffer = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * raw_samples
    
    pbar = tqdm(total=n_samples, desc="Running simulations Init Buffer", unit="sample")
    y_samples_buffer, y_names_buffer = system.run_simulation(x_samples_buffer, pbar=pbar)
    
    pbar.close()
    
    with h5py.File('inital_buffer.hdf5', 'w') as f:
        f.create_dataset('x_samples', data=x_samples_buffer)
        f.create_dataset('x_names', data=np.array(x_names, dtype='S'))
        
        f.create_dataset('y_samples', data=y_samples_buffer)
        f.create_dataset('y_names', data=np.array(y_names, dtype='S'))
    