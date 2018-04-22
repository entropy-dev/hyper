from training_all_autoencoders import main
import tensorflow as tf
import glob
import tqdm

#data_paths = ['/data/vis_data/2017_05_22/NoSpectralCorrection', '/data/vis_data/2017_05_22/WithSpectralCorrection']
data_paths = ['/tmp/data/NoS', '/tmp/data/WithS']
test_px = 0.1
network_shapes = [[16, 10, 10, 3], [15, 10, 10, 3]]
batch_size = 150000
epochs = 5
learning_rate = 0.0001

training_count = 5  # train each model 5 times
network_id_lut = {
    0: 'tied_weights',
    1: 'tied_weights_complex_loss',
    2: 'free_weights',
    3: 'free weights_complex_loss',
    4: 'free weights_complex_loss_densely_connected',
}

print('Starting\n\n\n')

for training_path, training_shape in tqdm.tqdm(list(zip(data_paths, network_shapes)), desc='Training Dataset'):
    for model_id, model_name in tqdm.tqdm(network_id_lut.items(), total=len(network_id_lut), desc='Mode id'):
        for training_idx in tqdm.trange(training_count, desc='Training index'):
            # individualization
            input_path = glob.glob("{}/*.hdf5".format(training_path))[0]
            log_path = '{}/results/{}/iteration_{}/logs/'.format(training_path, model_name, training_idx)
            model_path = '{}/results/{}/iteration_{}/models/model'.format(training_path, model_name, training_idx)

            main(data_path=input_path,
                 test_px=test_px,
                 network_shape=training_shape,
                 network_id=model_id,
                 batch_size=batch_size,
                 epochs=epochs,
                 learning_rate=learning_rate,
                 log_path=log_path,
                 model_path=model_path)
