import numpy as np
from functools import partial
from scipy.stats import norm
import warnings

class Prior:
    
    def __init__(self,
                 cs,
                 names,
                 types, 
                 pdfs,
                 sampling_funcs,
                 ranges,
                 modes,
                 values,
                 use_log_evaluation=True # not implemented yet
                ):
        # needed to be able to sample from distribution - used in the sample function
        self.prior_floor = 0


        # needed to be able to efficiently call the function - one per dimension, used in __call__
        self.cs = cs
        self.names = names
        self.types = types
        self.values = values
        self.pdfs = pdfs
        self.sampling_funcs = sampling_funcs
        self.ranges = ranges
        self.values = values    
        # TODO fix this, it scales wrong for integers
        self.mode = np.array(modes)#.reshape(1, -1)
        self.mode_non_numpy = modes
        self.dims = len(self.mode)
        self.max = self(self.mode.reshape(1, -1), normalize=False, normalized_input=False)
        # Should normalize sometimes, other times it should not...

    def get_pdfs(self):
        return self.evaluate_functions
    
    def sample(self, size, normalize=True, as_config=False):
        print('\n\n\nSAMPLING\n\n\n')
        oversampling_factor = 100
        samples = np.zeros((size*oversampling_factor, self.dims))

        for dim, func in enumerate(self.sampling_funcs):
            s = func(size=size*oversampling_factor)
            np.random.shuffle(s)
            samples[:, dim] = s

        in_bounds = np.array([True] * len(samples))
        
        # normalize samples to return values in 0, 1
        norm_samples = np.zeros(shape=(samples.shape))
        for dim, (range_, type_) in enumerate(zip(self.ranges, self.types)):
            if type_ != 'categorical':  
                lower, upper = range_
                param_in_bounds = (samples[:, dim] >= lower) & (samples[:, dim] <= upper)
                in_bounds = param_in_bounds & in_bounds
                norm_samples[:, dim] = (samples[:, dim] - lower) / (upper - lower) 
            else:
                norm_samples[:, dim] = samples[:, dim]
        
        if normalize:
            return_samples = norm_samples[in_bounds][0:size]
        else:
            return_samples = samples[in_bounds][0:size]

        if as_config:   
            return self._as_config(return_samples)
        else:
            return return_samples
    
    # TODO may need to be redefined for hypermapper - probably by providing it as dict?
    def _as_config(self, array):
        all_configs_list = []
        if len(array.shape) == 1:
            array = array.reshape(1, -1)
        for arr_config in array:
            
            config_dict = {}
            for i in range(len(arr_config)):
                    #print('\n\n\VALUES: ', self.values[i])
                    if self.types[i] == 'categorical':
                        config_dict[self.names[i]] = arr_config[i].astype(int)
                    elif self.types[i] == 'integer':
                        config_dict[self.names[i]] = self.values[i][np.round(arr_config[i]).astype(int)]
                    else:
                        config_dict[self.names[i]] = float(arr_config[i])
                    
            all_configs_list.append(config_dict)
        return all_configs_list
            

    def __call__(self, X, normalize=True, normalized_input=True):
        # everything comes in in range (0,1), and then gets scaled up to the proper range of the function
        X_scaled = np.zeros(X.shape)
        if normalized_input:
            for i, type_ in enumerate(self.types):
                if type_ == 'integer':
                    tot_numbers = self.ranges[i][1] - self.ranges[i][0] + 1
                    # since integers are input from SMAC with distance to the boundary
                    X_scaled[:, i] = (X[:, i] - 1/(2*tot_numbers)) * (tot_numbers) + self.ranges[i][0]

                else:
                    X_scaled[:, i] = X[:, i] * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
                
        else:
            X_scaled = X    
        
        # since several univariate distribution - compute across dimensions and return (assume independence between dims)
        probabilities = np.ones(len(X))
        # dimension-wise multiplication of the probabilities
        for i in range(self.dims):
            probabilities = probabilities * self.pdfs[i](X_scaled[:, i])
            
        if normalize:
             return probabilities.reshape(-1, 1) / self.max + self.prior_floor

        return probabilities.reshape(-1, 1)
    
    def get_max_location(self, as_config=False):
        if as_config:
            return self._as_config(self.mode)
        return self.mode_non_numpy

    def get_max(self):
        return self.max[0][0]
    
    def get_min(self):
        return self.prior_floor

    # only works for discrete - a quick fix
    def compute(self, configs):
        configs_numpy = np.array(configs)
        probs = np.zeros(configs_numpy.shape)

        for dim in range(self.dims):
            dim_probs = self.pdfs[dim]
            probs[:, dim] = dim_probs(configs_numpy[:, dim])

        return np.prod(probs, axis=1) / self.get_max()
        # TODO check the probabilities are correct, add zeros in the right spots and return probs
        

        

            



# takes a prior json as input and outputs all the necessary arguments for the prior
def process_prior(prior_file):
    
    def from_array(X, probs=None, range_offset=None):
        # if there are no values to consider, only return the element in the order
        # as with categoricals
        if range_offset is None:
            return probs[np.round(X).astype(int)]
        
        # otherwise, consider the value of the element and return the probability
        # in that spot
        else:
            indices = np.round(X).astype(int) - range_offset
            return probs[np.round(X).astype(int) - range_offset]
    
    names = []
    types = []
    pdfs = []
    sampling_funcs = []
    ranges = []
    modes = []
    values = []
    # needed to not cause memory issue with saved categorical probability arrays
    # this is sorted to prior keys as SMAC does the same for its configspace
    for key in prior_file.keys():
        names.append(key)
        #print('Prior input order: ', key)
        try:
            dist = prior_file[key]['dist']
        except KeyError:
            #print('Did not find distribution type, assuming gaussian')
            prior_file[key]['dist'] = 'gaussian'
        
        types.append(prior_file[key]['dist'])

        if prior_file[key]['dist'] == 'gaussian':
            values.append(None)
            mean = prior_file[key]['params']['mean'][0]
            std = prior_file[key]['params']['std'][0]
            ranges.append(prior_file[key]['range'])
            pdfs.append(norm(mean, std).pdf)
            sampling_funcs.append(norm(mean, std).rvs)
            modes.append(mean)
        
        # don't use the actual values, but an integer array with the same length
        # only used for sampling anyway  
        # give a range of 0,1 to prior so that it doesn't scale the input  
        elif prior_file[key]['dist'] == 'categorical':
            values.append(prior_file[key]['params']['values'])
            probs = np.array(prior_file[key]['params']['probs'])
            assert len(values[-1]) == len(probs),\
                f'{key} has values of length {len(values)}\nValues:{values}\nand probabilities of length{len(probs)}\nProbabilities{probs}'
                
            ranges.append([0, 1])
            modes.append(np.argmax(probs))
            pdfs.append(partial(from_array, probs=probs))
            sampling_funcs.append(partial(np.random.choice, len(probs), p=probs))
        
        # used for integer when you want to set the probablilities yourself 
        elif prior_file[key]['dist'] == 'integer':
            lower, upper = prior_file[key]['range'][0], prior_file[key]['range'][1]
            values.append(list(range(lower, upper+1)))
            probs = np.array(prior_file[key]['params']['probs'])
            assert len(values[-1]) == len(probs),\
                f'{key} has values of length {len(values)}\nValues:{values}\nand probabilities of length{len(probs)}\nProbabilities{probs}'
            ranges.append([lower, upper])
            modes.append(np.argmax(probs) + lower) 
            
            # need the scaling of the integer parameter
            # and the adjustment to the range
            # pdfs.append(lambda x: probs[np.round(x).astype(int)])
            pdfs.append(partial(from_array, probs=probs, range_offset=ranges[-1][0]))
            # and for the sampling, too
            sampling_funcs.append(partial(np.random.choice, len(probs), p=probs))

    ranges = np.array(ranges)
    return names, types, pdfs, sampling_funcs, ranges, modes, values

