from scipy.io import loadmat
from h5py import File as h5file
from os import listdir
from os.path import join
from numpy.random import permutation
from tqdm import tqdm
from math import ceil
import numpy as np
from argparse import ArgumentParser


def main(source_dir, destination_name, image_shape, blocksize, initial_run):
    if initial_run:
        print('Starting')

        filenames = [join(source_dir, name) for name in listdir(source_dir) if name.endswith('.mat')]

        print('Found {} files'.format(len(filenames)))

        total_pixel_count = len(filenames) * image_shape[0] * image_shape[1]
        destination_file = h5file(destination_name, mode='x', driver=None)
        dataset = destination_file.create_dataset(name='data',
                                                      shape=(total_pixel_count, image_shape[2]),
                                                      dtype='f',
                                                      data=None,
                                                      compression='gzip',
                                                      compression_opts=6)


        # Fill the dataset.
        image_pixel_count = image_shape[0] * image_shape[1]
        for i, filename in enumerate(tqdm(permutation(filenames))):
            data = loadmat(filename)['data'].reshape(image_pixel_count, image_shape[2])
            dataset[i*image_pixel_count:(i+1)*image_pixel_count] = permutation(data)


        # Save the dataset.
        dataset.flush()
    else:
        destination_file = h5file(destination_name, mode='r+', driver=None)
        dataset = destination_file['data']

    total_pixel_count = dataset.shape[0]

    # Shuffel the data.
    num_iterations = ceil(total_pixel_count/blocksize)

    for i in tqdm(range(num_iterations)):
        dataset[i*blocksize:(i+1)*blocksize] = permutation(np.array(dataset[i*blocksize:(i+1)*blocksize]))


    # Save all data and close files.
    dataset.flush()
    destination_file.close()

if __name__ == '__main__':
    command_help = 'This script creates a hyperspectral dataset.'
    parser = ArgumentParser(description=command_help)
    parser.add_argument('-s', '--source-dir', help='input folder of the raw images', required=True)
    parser.add_argument('-d', '--destination-name', help='output file', required=True)
    parser.add_argument('-i', '--image-shape', help='shape of the hyper cubes e.g. "214,407,25"', required=True)
    parser.add_argument('-b', '--blocksize', help='number of pixels to process at once e.g. 134217728', required=True)
    parser.add_argument('--no-files', dest='initial', default=True, action='store_false')
    args = vars(parser.parse_args())

    
    #source_dir = '/data/vis_data/2017_05_22/WithSpectralCorrection/PreprocessedImages/'
    #destination_name = '/data/vir_data/2017_05_22/WithSpectralCorrection/HyperspectralDataNir20170522.hdf5'
    image_shape = tuple([int(i) for i in args['image_shape'].split(',')])# (214, 407, 25) # (254, 510, 15)
    blocksize = int(args['blocksize']) # 1024*1024*1024 // 8 # 1GB * 15 * 2 ~ 30 GB

    main(args['source_dir'], args['destination_name'], image_shape, blocksize, args['initial'])