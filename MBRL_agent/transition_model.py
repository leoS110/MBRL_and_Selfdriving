import torch
import torch.nn as nn
from dataclasses import dataclass, field  

#from MBRL_train import train_params #problem
from config_train import TransitionConfig

nn_params = TransitionConfig()

#Define NN: forward pass
class pytorchNN(nn.Module):                             
    def __init__(self, dimension_in, dimension_out, n_width, n_layers, act=nn.ReLU):   
        super().__init__()                        # must run first: creates the parameter/module registries

        layers, d = [], dimension_in                      # layers accumulates modules; d tracks current width
        
        for _ in range(n_layers):                    #loop through each hidden layer
            layers += [nn.Linear(d, n_width), act()]   #linear mapping and then non-linear activation function
            d = n_width                            #reset the length of the tensor being passed at the input to the next layer

        layers.append(nn.Linear(d, dimension_out))        # output layer, no activation: targets are unbounded
        self.net = nn.Sequential(*layers)         # * unpacks the list into positional args; assigning registers it
 
    def forward(self, x):                         # runs on every batch; autograd records it as it executes
        return self.net(x)                        # Sequential applies each child module in order


#should this be part of the class?
def get_state_dif(model, state, action,  statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor):

    #everything in pytorch tensors

    #normalise model input
    x_val = torch.cat([state, action], dim=-1) #maybe should check dimensions of state & action
    x_val_normalised = (x_val - x_means_tensor)/x_stds_tensor
    model_val = model(x_val_normalised)

    #de-normalise model output 
    delta_s = model_val*statediff_stds_tensor + statediff_means_tensor

    return delta_s


#to define:
#transition_model = pytorchNN(nn_params.dimension_in, nn_params.dimension_out, nn_params.n_width, nn_params.n_layers)
#to eval s_t+1 = transition_model(s_t, a_t) or whatever the definition is

#loss function
#loss_fn = nn.MSELoss()   

#optimiser
#optimizer = torch.optim.SGD(transition_model.parameters(), lr=nn_params.lr) 

#torch.manual_seed(nn_params.seed)


#loss_array = np.empty((params.training_steps, 1), dtype=np.float64) #for loss curve plotting

#for loop_i in range(nn_params.training_steps):
#
#    optimizer.zero_grad(set_to_none=True) #zero the gradients from the last step so that they don't accumulate
#
#    #evaluate loss for loss curve, on full training set not on batch used for backprop, so don't use autograd to unnecessarily store gradient info
#    #with torch.no_grad():                        
#    #    lossval = loss_fn(model(x_TR), y_TR).item()
#    #    loss_array[loop_i] = lossval
#
#    #for SGD, generate a batch from training data:
#    idx = torch.randperm(dataset_size)[:(int(params.batch_proportion * dataset_size))]              # random batch, sampled without replacement
#    x_TR_batch, y_TR_batch = x_TR[idx], y_TR[idx]
#
#    #h(x), forward pass, autograd caching
#    h_val = model(x_TR_batch)  
#
#    #evaluate loss on batch
#    loss = loss_fn(h_val, y_TR_batch) 
#
#    #uses backprop + autograd to generate loss gradients (fills .grad graph)
#    loss.backward()
#
#    optimizer.step()        


