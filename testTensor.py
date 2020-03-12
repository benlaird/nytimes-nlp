from time import sleep

from tensorboard import program

import numpy as np
import tensorflow as tf
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter


tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

vectors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
metadata = ['Silly001', '010', '100', '111']  # labels
writer = SummaryWriter("Glove_vectors")
writer.add_embedding(vectors, metadata)
writer.close()
exit(0)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "/Users/rebjl2/PycharmProjects/nytimes-nlp/runs"])
url = tb.launch()

print('TensorBoard at %s \n' % url)
sleep(10)
