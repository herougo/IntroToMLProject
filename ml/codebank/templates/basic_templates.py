'''
This gives templates for functions so they can make use of all the codebank helper functions.
It is recommended to file the same format as in the examples, but researchers have their own
styles, so this is meant to be flexible.
'''
def get_cifar_datasets(root, download=False):
    ...
    return {'train': train_dataset, 'test': test_dataset}

class BasicClassificationTrainer:
    '''
    TRAINING_DICT_ATTRS and STATE_DICT_ATTRS are used by helper functions for
    determining attribute access for checkpointing
    '''
    TRAINING_DICT_ATTRS = 'current_epoch'.split()
    STATE_DICT_ATTRS = 'model models optimizer optimizers'.split()