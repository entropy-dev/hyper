
# coding: utf-8
'''
Hyperspectral Dataset Streamer

This file contains a loader and wrapper class for hyperspectral datasets in hdf5
format. It is used to provide access to datasets that are bigger than the
available memory. It is ensured that each batch is exactly of size batchsize.
'''


from h5py import File as _file

class HyperspectralDatasetStreamer:
    """
    A wrapper class for autoencoder datasets.
    """

    def __init__(self, path, test_px, dataset_id='data'):
        """
        :param path:       path to the dataset.
        :param test_px:    size of the testing data
        :param dataset_id: dataset to load from the database
        """
        self._f = _file(path,'r')
        self._data = self._f[dataset_id]

        size = self._data.shape[0]

        self._test_start = 0
        self._test_end = int(size*test_px)
        self._train_start = self._test_end
        self._train_end = size

    def train_epoch(self, batchsize):
        yield from self._epoch(batchsize, self._train_start, self._train_end)

    def test_epoch(self, batchsize):
        yield from self._epoch(batchsize, self._test_start, self._test_end)

    def _epoch(self, batchsize, start, end):
        iterations = (end-start) // batchsize

        if not iterations:
            raise RuntimeError('Batchsize is to big to get at least one full batch!')

        for i in range(iterations):
            idx_0 = start + batchsize * i
            idx_1 = start + batchsize * (i + 1)
            yield self._data[idx_0:idx_1]
    
    def get_train_size():
        return self._train_end - self._train_start

    def get_test_size():
        return self._test_end - self._test_start
