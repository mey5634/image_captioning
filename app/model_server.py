import tensorflow as tf
import sys, os
CURDIR = os.path.dirname(os.path.realpath(__file__))
ROOTDIR = os.path.abspath(os.path.join(CURDIR, '..'))
sys.path.insert(0,ROOTDIR)

from config import Config
from model import CaptionGenerator
from dataset import build_vocabulary
from utils.vocabulary import Vocabulary


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'test',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', True,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', os.path.join(ROOTDIR,'models','289999.npy'),
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')


class ModelServer:
    def __init__(self):
        curdir = os.getcwd()
        os.chdir(ROOTDIR)

        self.config = Config()
        self.config.phase = FLAGS.phase
        self.config.train_cnn = FLAGS.train_cnn
        self.config.beam_size = FLAGS.beam_size

        print("Building the vocabulary...")
        if os.path.exists(self.config.vocabulary_file):
            self.vocabulary = Vocabulary(self.config.vocabulary_size,
                                         self.config.vocabulary_file)
        else:
            self.vocabulary = build_vocabulary(self.config)
        print("Vocabulary built.")
        print("Number of words = %d" %(self.vocabulary.size))

        self.sess = tf.Session()

        self.model = CaptionGenerator(self.config)
        self.model.load(self.sess, FLAGS.model_file)
        tf.get_default_graph().finalize()

        os.chdir(curdir)

if __name__ == "__main__":
    model = ModelServer()
    print('done!')