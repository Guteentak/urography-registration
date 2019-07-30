# python imports
import os

from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# project imports
import networks
import losses


def resize_depth(data, std_depth=128):
    # data : npz file
    wide, height, depth = data.shape
    diff = std_depth - depth
    term = std_depth / diff
    string = np.arange(1,std_depth-1,term)
    string = string.astype(int)

    for i in string:
        tmp = np.empty((wide, height))
        tmp[:] = np.nan
        data = np.insert(data, i, tmp, axis=2)

    for i in np.arange(data.shape[2]):
        if np.isnan(data[0, 0, i]):
            data[:, :, i] = (data[:, :, i-1] + data[:, :, i+1]) / 2

    return data


# add depth resize code
def load_data(data_dir, depth_size, mode='train', fixed='first'):
    path = os.path.join(data_dir, mode)
    volumes = [np.load(os.path.join(path, filename)) for filename in os.listdir(path) if '.npz' in filename]

    moving_img = []
    fixed_img = []

    for volume in volumes:
        first = volume['first']
        second = volume['second']
        if fixed == 'first':
            moving_img.append(resize_depth(second)[np.newaxis, ..., np.newaxis])
            fixed_img.append(resize_depth(first)[np.newaxis, ..., np.newaxis])
        else:
            moving_img.append(resize_depth(first)[np.newaxis, ..., np.newaxis])
            fixed_img.append(resize_depth(second)[np.newaxis, ..., np.newaxis])
    moving_img = np.concatenate(moving_img, axis=0)  # (number of data, w, h, depth, channel)
    fixed_img = np.concatenate(fixed_img, axis=0)

    zeros = np.zeros(shape=fixed_img.shape)

    return [moving_img, fixed_img], [fixed_img, zeros]


def train(data_dir,
          depth_size,
          model,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          reg_param,
          steps_per_epoch,
          train_mode,
          load_model_file,
          data_loss,
	  batch_size,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc'
    """

    # load data
    train_x, train_y = load_data(data_dir=data_dir, depth_size=depth_size, mode='train', fixed='first')
#    test_x, test_y = load_data(data_dir=data_dir, depth_size=depth_size, mode='test', fixed='first')

    vol_size = train_x[0].shape[1:-1]  # (width, height, depth)

    # set encoder, decoder feature number
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:  # 'vm2double':
        nf_enc = [f * 2 for f in nf_enc]
        nf_dec = [f * 2 for f in [32, 32, 32, 32, 32, 16, 16]]

    # set loss function
    # Mean Squared Error, Cross-Correlation, Negative Cross-Correlation
    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        data_loss = losses.NCC().loss

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    gpu = '/gpu:%d' % 0  # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        # in the CVPR layout, the model takes in [moving image, fixed image] and outputs [warped image, flow]
        model = networks.cvpr2018_net(tuple(vol_size), nf_enc, nf_dec)

        model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # fit
    with tf.device(gpu):
        save_callback = ModelCheckpoint(save_file_name)
        mg_model = model

        # compile
        mg_model.compile(optimizer=Adam(lr=lr),
                         loss=[data_loss, losses.Grad('l2').loss],
                         loss_weights=[1.0, reg_param])

        # fit
        mg_model.fit(x=train_x,
                     y=train_y,
                     batch_size=None,
                     epochs=nb_epochs,
                     verbose=1,
                     callbacks=[save_callback],
                     steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        help="data folder")
    parser.add_argument("--depth_size", type=int,
                        dest="depth_size", default=128,
                        help="(width, height, depth) must be int")
    parser.add_argument("--model", type=str, dest="model",
                        choices=['vm1', 'vm2', 'vm2double'], default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/first_try/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default=7,
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--train_mode", type=str,
                        dest="train_mode", default='transfer',
                        help=
                        """
                        train mode
                        - 'transfer': using cvpr2018 model weight.
                        - 'continue': using trained model weight.
                        """)
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='../models/cvpr2018_vm2_l2.h5',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--data_loss", type=str,
                        dest="data_loss", default='mse',
                        help="data_loss: mse of ncc")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=None,
                        help="batch size. default 'None'.")

    args = parser.parse_args()
    train(**vars(args))
