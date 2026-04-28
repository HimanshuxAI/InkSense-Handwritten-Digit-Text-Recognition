"""
Handwritten Text Recognition (HTR) Model.
Based on: https://github.com/githubharald/SimpleHTR

Uses CNN + RNN + CTC architecture implemented with TensorFlow
to recognize handwritten text from images.

Modified to use an isolated TF1 graph/session so it doesn't conflict
with the Keras-based digit recognition model.
"""

import os
import sys
from typing import List, Tuple

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import tensorflow as tf


class DecoderType:
    """CTC decoder types."""
    BestPath = 0
    BeamSearch = 1


class HTRModel:
    """Minimalistic TF model for HTR using isolated graph."""

    def __init__(self, char_list, decoder_type=DecoderType.BestPath,
                 must_restore=False, model_dir=None):
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'models', 'htr_model')

        # Create isolated graph and session
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input image batch
            self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
            self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

            # setup CNN, RNN and CTC
            self.setup_cnn()
            self.setup_rnn()
            self.setup_ctc()

            # initialize TF
            self.sess = tf.compat.v1.Session(graph=self.graph)
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

            latest_snapshot = tf.train.latest_checkpoint(self.model_dir)
            if self.must_restore and not latest_snapshot:
                raise Exception('No saved model found in: ' + self.model_dir)

            if latest_snapshot:
                print('Init HTR with stored values from ' + latest_snapshot)
                self.saver.restore(self.sess, latest_snapshot)
            else:
                print('Init HTR with new values')
                self.sess.run(tf.compat.v1.global_variables_initializer())

    def setup_cnn(self):
        """Create CNN layers."""
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis=3)

        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        pool = cnn_in4d
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.random.truncated_normal(
                    [kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]],
                    stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(
                input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')

        self.cnn_out_4d = pool

    def setup_rnn(self):
        """Create RNN layers."""
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
                 for _ in range(2)]
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d, dtype=rnn_in3d.dtype)
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(
            tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setup_ctc(self):
        """Create CTC loss and decoder."""
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])

        self.gt_texts = tf.SparseTensor(
            tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
            tf.compat.v1.placeholder(tf.int32, [None]),
            tf.compat.v1.placeholder(tf.int64, [2]))

        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])

        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(
                labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,
                sequence_length=self.seq_len, ctc_merge_repeated=True))

        self.saved_ctc_input = tf.compat.v1.placeholder(
            tf.float32, shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(
            labels=self.gt_texts, inputs=self.saved_ctc_input,
            sequence_length=self.seq_len, ctc_merge_repeated=True)

        if self.decoder_type == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(
                inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)
        elif self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(
                inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len, beam_width=50)

    def to_sparse(self, texts):
        indices = []
        values = []
        shape = [len(texts), 0]
        for batchElement, text in enumerate(texts):
            label_str = [self.char_list.index(c) for c in text]
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)
        return indices, values, shape

    def decoder_output_to_text(self, ctc_output, batch_size):
        decoded = ctc_output[0][0]
        label_strs = [[] for _ in range(batch_size)]
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]
            label_strs[batch_element].append(label)
        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]

    def infer_batch(self, imgs, calc_probability=False):
        """Feed a batch into the NN to recognize the texts."""
        num_batch_elements = len(imgs)
        eval_list = [self.decoder]

        if calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        max_text_len = imgs[0].shape[0] // 4

        feed_dict = {
            self.input_imgs: imgs,
            self.seq_len: [max_text_len] * num_batch_elements,
            self.is_train: False
        }

        eval_res = self.sess.run(eval_list, feed_dict)
        decoded = eval_res[0]
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        probs = None
        if calc_probability:
            sparse = self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {
                self.saved_ctc_input: ctc_input,
                self.gt_texts: sparse,
                self.seq_len: [max_text_len] * num_batch_elements,
                self.is_train: False
            }
            loss_vals = self.sess.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        return texts, probs


class TextRecognizer:
    """High-level wrapper for text recognition inference."""

    def __init__(self):
        self.model = None
        self.char_list = None
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'models', 'htr_model')
        self._initialized = False
        self._init_error = None

        try:
            self._load_model()
        except Exception as e:
            self._init_error = str(e)
            print(f"[WARNING] HTR model not loaded: {e}")

    def _load_model(self):
        """Load the pre-trained HTR model."""
        char_list_path = os.path.join(self.model_dir, 'charList.txt')
        if not os.path.exists(char_list_path):
            raise FileNotFoundError(
                f"Character list not found at {char_list_path}. "
                "Download the pre-trained model — see README.md."
            )

        with open(char_list_path) as f:
            self.char_list = list(f.read())

        self.model = HTRModel(
            self.char_list, DecoderType.BestPath,
            must_restore=True, model_dir=self.model_dir)
        self._initialized = True
        print("[INFO] HTR model loaded successfully.")

    def process_img(self, img, img_size=(128, 32)):
        """Preprocess image for HTR model."""
        if img is None:
            img = np.zeros(img_size[::-1])

        img = img.astype(np.float64)

        # Resize keeping aspect ratio
        wt, ht = img_size
        h, w = img.shape
        f = min(wt / w, ht / h)
        tx = (wt - w * f) / 2
        ty = (ht - h * f) / 2

        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones([ht, wt]) * 255
        img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target,
                             borderMode=cv2.BORDER_TRANSPARENT)

        # Transpose for TF
        img = cv2.transpose(img)

        # Normalize to [-0.5, 0.5]
        img = img / 255 - 0.5
        return img

    def predict(self, image, line_mode=False):
        """Recognize text in a handwritten image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if not self._initialized:
            return {
                'text': '[Model not loaded - see README for setup instructions]',
                'probability': 0.0,
                'error': self._init_error
            }

        img_size = (256, 32) if line_mode else (128, 32)
        processed = self.process_img(image, img_size)

        texts, probs = self.model.infer_batch([processed], calc_probability=True)

        return {
            'text': texts[0],
            'probability': float(probs[0]) if probs is not None else 0.0
        }
