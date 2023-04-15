from spikingjelly.activation_based import surrogate

class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    datasets_path = '../Datasets'

    seed = 0

    model_type = 'snn_delays'          # 'ann', 'snn', 'snn_delays' 'snn_delays_lr0'
    dataset = 'shd'             # 'shd', 'ssc'

    time_step = 10
    n_bins = 10

    epochs = 50
    batch_size = 128

    rnoise_sig = 0

    ################################################
    #               Model Achitecture              #
    ################################################
    spiking_neuron_type = 'plif'         # plif, lif
    init_tau = 15.0                      # in ms, can't be < time_step

    stateful_synapse_tau = 15.0         # in ms, can't be < time_step
    stateful_synapse = True
    stateful_synapse_learnable = True

    n_inputs = 700//n_bins
    n_hidden_layers = 2
    n_hidden_neurons = 128
    n_outputs = 20 if dataset == 'shd' else 35
    
    dropout_p = 0.2
    use_batchnorm = True
    bias = False
    detach_reset = True

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum
    loss_fn = 'CEloss'
    output_v_threshold = 2.0 if loss == 'spike_count' else 1e9  #use 1e9 for loss = 'mean' or 'max'

    v_threshold = 1.0
    alpha = 5.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)

    init_w_method = 'kaiming_uniform'

    init_tau = (init_tau  +  1e-9) / time_step
    stateful_synapse_tau = (stateful_synapse_tau  +  1e-9) / time_step
    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    lr_w = 1e-3
    lr_pos = 200*lr_w   if model_type =='snn_delays' else 0
    
    # 'one_cycle', 'cosine_a', 'none'
    scheduler_w = 'one_cycle'    
    scheduler_pos = 'cosine_a'   if model_type =='snn_delays' else 'none'


    # for one cycle
    max_lr_w = 5 * lr_w
    max_lr_pos = 5 * lr_pos


    # for cosine annealing
    t_max_w = epochs
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'gauss'
    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 250//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
    
    sigInit = max_delay // 4
    final_epoch = (1*epochs)//2


    left_padding = max_delay-1
    right_padding = (max_delay-1) // 3

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2

    #############################
    #           Wandb           #
    #############################
    use_wandb = True

    wandb_project_name = 'SHD-BestACC'
    wandb_run_name = f'Baseline Tests||Ts = 10ms, Learnable taus||{dataset}||{model_type}||{loss}||MaxDelay={max_delay}||neuron={spiking_neuron_type}||seed={seed}'

    wandb_group_name = f"Baseline Tests {model_type}"


