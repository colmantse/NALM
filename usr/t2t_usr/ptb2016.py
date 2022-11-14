# coding=utf-8
# Copyright 2022 The NALM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for PTB 2016 (Schmaltz et al. 2016) processed version
   and our ngram based permutation versions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.utils import mlperf_log
from tensor2tensor.data_generators import ptb
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils.t2t_model import log_info

import os
import tensorflow as tf

def ptb_line_iterator(txt_path):
  with tf.gfile.GFile(txt_path, "r") as f:
    for line in f:
      line = " ".join(line.replace("\n", " %s " % "").split())
      yield line

def ptb_shuffle_iterator(ground_truth_txt_path,target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  log_info("running custom iterator...")
  for inputs, targets in zip(
      ptb_line_iterator(ground_truth_txt_path), ptb_line_iterator(target_txt_path)):
    yield {"ground_truth": inputs, "targets": targets}

def ptb_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True,
                               inputs_prefix="",
                               targets_prefix=""):
  """Encode Text2Text samples from the generator with the vocab."""
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    if has_inputs:
      sample["inputs"] = vocab.encode(inputs_prefix + sample["inputs"])
      sample["inputs"].append(text_encoder.EOS_ID)
    sample["targets"] = targets_vocab.encode(targets_prefix + sample["targets"])
    sample["targets"].append(text_encoder.EOS_ID)
    if "ground_truth" in sample:
      sample["ground_truth"] = vocab.encode(targets_prefix + sample["ground_truth"])
      sample["ground_truth"].append(text_encoder.EOS_ID)
      if len(sample["ground_truth"]) != len(sample["targets"]):
        continue
    yield sample


@registry.register_problem
class Ptb2016Reorder(ptb.LanguagemodelPtb10k):
  """No eos token for this problem
     2016 Allen schutze version"""
  @property
  def vocab_filename(self):
    return "vocab.ptb2016.tokens"

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    filename = 'ptb2016'
    tf.gfile.MakeDirs(tmp_dir)
    filepath = os.path.join(tmp_dir, filename)
    if not tf.gfile.Exists(filepath):
      raise FileNotFoundError('ptb2016 cannot be found')
    tf.logging.info("folder found: %s" % filepath)
    files = [filename+'/'+file for file in os.listdir(filepath)]

    train_file, valid_file = None, None
    for filename in files:
      if "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"

    ptb._get_token_encoder(data_dir, self.vocab_filename, train_file)

    if dataset_split == problem.DatasetSplit.TRAIN:
      filepath = train_file
    elif dataset_split == problem.DatasetSplit.EVAL:
      filepath = valid_file

    def _generate_samples():
      with tf.gfile.GFile(filepath, "r") as f:
        for line in f:
          line = " ".join(line.replace("\n", " %s " % "").split())
          yield {"targets": line}

    return _generate_samples()

@registry.register_problem
class Ptb2016Shuffle(Ptb2016Reorder):
  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
    encoders = {"targets": encoder, "ground_truth": encoder}
    if self.has_inputs:
      encoders["inputs"] = encoder
    return encoders

  def generate_samples(self, data_dir, tmp_dir, dataset_split, custom_iterator=ptb_shuffle_iterator):
    filename = 'ptb2016'
    tf.gfile.MakeDirs(tmp_dir)
    filepath = os.path.join(tmp_dir, filename)
    if not tf.gfile.Exists(filepath):
      raise FileNotFoundError('ptb2016 cannot be found')
    tf.logging.info("folder found: %s" % filepath)
    files = [filename+'/'+file for file in os.listdir(filepath)]

    train_file, valid_file = None, None
    for filename in files:
      if "train.fullyshuffled" in filename:
        shuff_train = os.path.join(tmp_dir,filename)
      elif "valid.fullyshuffled" in filename:
        shuff_valid = os.path.join(tmp_dir,filename)
      elif "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"
    assert shuff_train, "Shuffling Training file not found"
    assert shuff_valid, "Shuffling Validation file not found"

    ptb._get_token_encoder(data_dir, self.vocab_filename, train_file)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filenames = (train_file,shuff_train)
    elif dataset_split == problem.DatasetSplit.EVAL:
      filenames = (valid_file,shuff_valid)

    return custom_iterator(filenames[0],filenames[1])

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    elif dataset_split == problem.DatasetSplit.EVAL:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)
    log_info('Generating '+dataset_split)

    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return ptb_generate_encoded(generator, encoder,
                                      has_inputs=self.has_inputs,
                                      inputs_prefix=self.inputs_prefix,
                                      targets_prefix=self.targets_prefix)

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (super(Ptb2016Shuffle,
                                                 self)
                                           .example_reading_spec())
    data_fields["ground_truth"] = tf.VarLenFeature(tf.int64)
    return (data_fields, data_items_to_decoders)

@registry.register_problem
class Ptb2016ShuffleR04p2(Ptb2016Shuffle):
  def generate_samples(self, data_dir, tmp_dir, dataset_split, 
                       custom_iterator=ptb_shuffle_iterator):
    filename = 'ptb2016'
    tf.gfile.MakeDirs(tmp_dir)
    filepath = os.path.join(tmp_dir, filename)
    if not tf.gfile.Exists(filepath):
      raise FileNotFoundError('ptb2016 cannot be found')
    tf.logging.info("folder found: %s" % filepath)
    files = [filename+'/'+file for file in os.listdir(filepath)]

    train_file, valid_file = None, None
    for filename in files:
      if "train.shuffle.0.4_p2" in filename:
        shuff_train = os.path.join(tmp_dir,filename)
      elif "valid.shuffle.0.4_p2" in filename:
        shuff_valid = os.path.join(tmp_dir,filename)
      elif "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"
    assert shuff_train, "Shuffling Training file not found"
    assert shuff_valid, "Shuffling Validation file not found"

    ptb._get_token_encoder(data_dir, self.vocab_filename, train_file)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filenames = (train_file,shuff_train)
    elif dataset_split == problem.DatasetSplit.EVAL:
      filenames = (valid_file,shuff_valid)

    return custom_iterator(filenames[0],filenames[1])

@registry.register_problem
class Ptb2016ShuffleR06p2(Ptb2016Shuffle):
  def generate_samples(self, data_dir, tmp_dir, dataset_split, custom_iterator=ptb_shuffle_iterator):
    filename = 'ptb2016'
    tf.gfile.MakeDirs(tmp_dir)
    filepath = os.path.join(tmp_dir, filename)
    if not tf.gfile.Exists(filepath):
      raise FileNotFoundError('ptb2016 cannot be found')
    tf.logging.info("folder found: %s" % filepath)
    files = [filename+'/'+file for file in os.listdir(filepath)]

    train_file, valid_file = None, None
    for filename in files:
      if "train.shuffle.0.6_p2" in filename:
        shuff_train = os.path.join(tmp_dir,filename)
      elif "valid.shuffle.0.6_p2" in filename:
        shuff_valid = os.path.join(tmp_dir,filename)
      elif "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"
    assert shuff_train, "Shuffling Training file not found"
    assert shuff_valid, "Shuffling Validation file not found"

    ptb._get_token_encoder(data_dir, self.vocab_filename, train_file)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filenames = (train_file,shuff_train)
    elif dataset_split == problem.DatasetSplit.EVAL:
      filenames = (valid_file,shuff_valid)

    return custom_iterator(filenames[0],filenames[1])

@registry.register_problem
class Ptb2016ShuffleR08p2(Ptb2016Shuffle):
  def generate_samples(self, data_dir, tmp_dir, dataset_split, custom_iterator=ptb_shuffle_iterator):
    filename = 'ptb2016'
    tf.gfile.MakeDirs(tmp_dir)
    filepath = os.path.join(tmp_dir, filename)
    if not tf.gfile.Exists(filepath):
      raise FileNotFoundError('ptb2016 cannot be found')
    tf.logging.info("folder found: %s" % filepath)
    files = [filename+'/'+file for file in os.listdir(filepath)]

    train_file, valid_file = None, None
    for filename in files:
      if "train.shuffle.0.8_p2" in filename:
        shuff_train = os.path.join(tmp_dir,filename)
      elif "valid.shuffle.0.8_p2" in filename:
        shuff_valid = os.path.join(tmp_dir,filename)
      elif "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"
    assert shuff_train, "Shuffling Training file not found"
    assert shuff_valid, "Shuffling Validation file not found"

    ptb._get_token_encoder(data_dir, self.vocab_filename, train_file)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filenames = (train_file,shuff_train)
    elif dataset_split == problem.DatasetSplit.EVAL:
      filenames = (valid_file,shuff_valid)

    return custom_iterator(filenames[0],filenames[1])

@registry.register_problem
class Ptb2016DisplaceR05p2(Ptb2016Shuffle):
  def generate_samples(self, data_dir, tmp_dir, dataset_split, custom_iterator=ptb_shuffle_iterator):
    filename = 'ptb2016'
    tf.gfile.MakeDirs(tmp_dir)
    filepath = os.path.join(tmp_dir, filename)
    if not tf.gfile.Exists(filepath):
      raise FileNotFoundError('ptb2016 cannot be found')
    tf.logging.info("folder found: %s" % filepath)
    files = [filename+'/'+file for file in os.listdir(filepath)]

    train_file, valid_file = None, None
    for filename in files:
      if "train.displace.0.5_p2" in filename:
        shuff_train = os.path.join(tmp_dir,filename)
      elif "valid.displace.0.5_p2" in filename:
        shuff_valid = os.path.join(tmp_dir,filename)
      elif "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"
    assert shuff_train, "Shuffling Training file not found"
    assert shuff_valid, "Shuffling Validation file not found"

    ptb._get_token_encoder(data_dir, self.vocab_filename, train_file)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filenames = (train_file,shuff_train)
    elif dataset_split == problem.DatasetSplit.EVAL:
      filenames = (valid_file,shuff_valid)

    return custom_iterator(filenames[0],filenames[1])


