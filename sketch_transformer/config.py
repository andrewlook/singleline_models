from .utils import CN


def get_default_config():
    C = CN()
    C.n_layer = 4
    C.n_head = 8
    C.d_model =  128
    C.d_ff = 512
    C.d_lowerdim = 256

    # these options must be filled in externally
    C.vocab_size = None
    C.block_size = None
    # dropout hyperparameters
    C.dropout_rate = 0.1


    C.max_seq_length = 250
    C.batch_size = 100

    C.blind_decoder_mask = True # if True, the decoder knows padding location of the input

    # TODO: just make this a path?
    C.dataset_source = 'look'
    C.dataset_name = 'epoch20240221_expanded10x_trainval'

    C.epochs = 50000
    C.lr = 1e-3
    C.use_lr_decay = True
    C.min_lr = 1e-5
    C.lr_decay = 0.9999
    
    return C

hp = get_default_config()