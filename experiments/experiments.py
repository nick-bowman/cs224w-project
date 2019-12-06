from gcn_experiments import run_experiment

import torch

def main():
    args_list = [
        {
            'model_type': 'GCN',
            'num_layers': 3, 
            'hidden_dim': 128,
            'dropout': 0.1, 
            'epochs': 300,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'weight_decay': 5e-3,
            'lr': 0.01
        },
        {
            'model_type': 'GraphSage',
            'num_layers': 3, 
            'hidden_dim': 64,
            'dropout': 0.5, 
            'epochs': 300,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'weight_decay': 5e-3,
            'lr': 0.01
        },
        {
            'model_type': 'GAT',
            'num_heads': 2,
            'num_layers': 3, 
            'hidden_dim': 64,
            'dropout': 0.05, 
            'epochs': 1000,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'weight_decay': 5e-3,
            'lr': 0.01
        }  
    ]
    run_experiment("es", 2005, 2006, args_list, feature_dir="Spanish2005")
#     run_experiment("es", 2005, 2006, args_list)
#     run_experiment("ru", 2005, 2006, args_list)
#     run_experiment("en", 2002, 2003, args_list)
#     run_experiment("de", 2004, 2005, args_list)
#     run_experiment("fr", 2004, 2005, args_list)
#     run_experiment("it", 2005, 2006, args_list)
#     run_experiment("nl", 2005, 2006, args_list)

if __name__ == "__main__":
    main()