
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DO NOT CHANGE THE STRUCTURE OF THE DICTIONARY. 

configs = {
    
    'Hopper-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 32,
                'n_layers': 3,
                'batch_size': None,
                'env_type': 'Hopper-v4'
            },
            "num_iteration": 100,
            "episode_len": 5000
    },
    
    
    'Ant-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 64,
                'n_layers': 2,
                'batch_size': None,
                'env_type': 'Ant-v4' 
            },
            "num_iteration": 150,
            "episode_len": 5000
    }

}