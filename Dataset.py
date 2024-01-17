import numpy as np
class Dataset:

    def __init__(self, x, y, batch_size = 32):

        self.x = x
        self.y = y

        self.batch_size = batch_size

        self.length = int(np.ceil(x.shape[0]/batch_size))

        self.indices = np.arange(x.shape[0])

    def __getitem__(self, i):

        i0 = i*self.batch_size
        i1 = min((i + 1)*self.batch_size, self.x.shape[0])

        index = self.indices[i0:i1]

        return self.x[index], self.y[index]

    def __len__(self):
        return self.length

    def shuffle(self):
        self.indices = np.random.permutation(self.indices)