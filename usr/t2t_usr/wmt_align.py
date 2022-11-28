from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import translate_ende
from tensor2tensor.data_generators import problem 
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems

from tensor2tensor.utils import registry
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import contrib
from tensor2tensor.utils.t2t_model import log_info

import os
import tensorflow as tf


_ENDE_TRAIN_DATASETS = [
    [
        "https://www.statmt.org/wmt15/training-parallel-nc-v10.tgz",  # pylint: disable=line-too-long
        ("training-parallel-nc-v10/news-commentary-v10.de-en.en",
         "training-parallel-nc-v10/news-commentary-v10.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.de-en.en", "commoncrawl.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.de-en.en", "training/europarl-v7.de-en.de")
    ],
]

_ENDE_TEST_DATASETS = [
    [
        "",
        ("dev/newstest2015-ende-src.en.sgm", "dev/newstest2015-ende-ref.de.sgm")
    ],
]

_JAEN_TRAIN_DATASETS = [
    ["http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/2.0/bitext/en-ja.tar.gz",
     ('tsv',3,2,'en-ja/en-ja.bicleaner05.txt')
    ],
    ["https://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-ja.tsv.gz",
    ('tsv',1,0,'news-commentary-v15.en-ja.tsv')
    ],
    ["https://nlp.stanford.edu/projects/jesc/data/split.tar.gz",
    ('tsv',1,0,'split/train')
    ],
    ["http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz",
    ('kftt-data-1.0/data/orig/kyoto-train.ja','kftt-data-1.0/data/orig/kyoto-train.en')
    ],
]

_JAEN_EVAL_DATASETS = [
    ["http://data.statmt.org/wmt20/translation-task/dev.tgz",
     ("dev/newsdev2020-jaen-src.ja.sgm","dev/newsdev2020-jaen-ref.en.sgm")
    ],
]

_JAEN_TEST_DATASETS = [
    ["http://data.statmt.org/wmt20/translation-task/test-ts.tgz",
     ("sgm/newstest2020-jaen-src-ts.ja.sgm","sgm/newstest2020-jaen-ref-ts.en.sgm")
    ],
]

@registry.register_problem
class WmtDeenReorder(translate_ende.TranslateEndeWmt32k):
  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  @property
  def approx_vocab_size(self):
    return 50000  

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def source_data_files(self, dataset_split):
    train_datasets = _ENDE_TRAIN_DATASETS + self.additional_training_datasets
    if dataset_split == problem.DatasetSplit.TRAIN:
      return train_datasets
    elif dataset_split == problem.DatasetSplit.EVAL:
      return translate_ende._ENDE_EVAL_DATASETS
    else:
      return _ENDE_TEST_DATASETS

  def generate_samples(
      self,
      data_dir,
      tmp_dir,
      dataset_split,
      custom_iterator=text_problems.text2text_txt_iterator):
    datasets = self.source_data_files(dataset_split)
    tag = "test"
    datatypes_to_clean = None
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = "train"
      datatypes_to_clean = self.datatypes_to_clean
    elif dataset_split == problem.DatasetSplit.EVAL:
      tag = "dev"
    data_path = translate.compile_data(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag),
        datatypes_to_clean=datatypes_to_clean)

    return custom_iterator(data_path + ".lang1", data_path + ".lang2")

  def vocab_data_files(self):
    return [['',("%s-compiled-%s" % (self.name, "train") + ".lang1",
                 "%s-compiled-%s" % (self.name, "train") + ".lang2")]]
   
  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    elif dataset_split == problem.DatasetSplit.EVAL:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)
    log_info('Generating '+dataset_split)
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return text2text_generate_encoded(generator, encoder,
                                      has_inputs=self.has_inputs,
                                      inputs_prefix=self.inputs_prefix,
                                      targets_prefix=self.targets_prefix)

  def input_fn(self,
               mode,
               hparams,
               data_dir=None,
               params=None,
               config=None,
               force_repeat=False,
               prevent_repeat=False,
               dataset_kwargs=None):
    """Builds input pipeline for problem.

    Args:
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hparams
      data_dir: str, data directory; if None, will use hparams.data_dir
      params: dict, may include "batch_size"
      config: RunConfig; should have the data_parallelism attribute if not using
        TPU
      force_repeat: bool, whether to repeat the data even if not training
      prevent_repeat: bool, whether to not repeat when in training mode.
        Overrides force_repeat.
      dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
        method when called

    Returns:
      (features_dict<str name, Tensor feature>, Tensor targets)
    """
    partition_id, num_partitions = self._dataset_partition(mode, config, params)
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if config and config.use_tpu:
      num_threads = 64
    else:
      num_threads = data_reader.cpu_count() if is_training else 1
    data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)
    dataset_kwargs = dataset_kwargs or {}
    dataset_kwargs.update({
        "mode": mode,
        "data_dir": data_dir,
        "num_threads": num_threads,
        "hparams": hparams,
        "partition_id": partition_id,
        "num_partitions": num_partitions,
    })
    return reorder_input_fn(
        self.dataset(**dataset_kwargs),
        self.filepattern(data_dir, mode),
        self.skip_random_fraction_when_training,
        self.batch_size_means_tokens,
        self.get_hparams().batch_size_multiplier,
        self.max_length(hparams),
        mode,
        hparams,
        data_dir=data_dir,
        params=params,
        config=config,
        force_repeat=force_repeat,
        prevent_repeat=prevent_repeat)

@registry.register_problem
class WmtDeenReorder32k(WmtDeenReorder):
  @property
  def approx_vocab_size(self):
    return 2**15

@registry.register_problem
class WmtDeenReorderWord(WmtDeenReorder):
  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def oov_token(self):
    return "<UNK>"

  @property
  def vocab_filename(self):
    return "vocab.wmt_reorder.50000.tokens"

@registry.register_problem
class WmtJaenReorder32k(WmtDeenReorder32k):
  def source_data_files(self, dataset_split):
    train_datasets = _JAEN_TRAIN_DATASETS + self.additional_training_datasets
    if dataset_split == problem.DatasetSplit.TRAIN:
      return train_datasets
    elif dataset_split == problem.DatasetSplit.EVAL:
      return _JAEN_EVAL_DATASETS
    else:
      return _JAEN_TEST_DATASETS

def compile_aligned_data(tmp_dir, datasets, filename, datatypes_to_clean=None):
  """Concatenates all `datasets` and saves to `filename`."""
  datatypes_to_clean = datatypes_to_clean or []
  filename = os.path.join(tmp_dir, filename)
  lang1_fname = filename + ".lang1"
  lang2_fname = filename + ".lang2"
  lang2a_fname = filename + ".lang2a"
  if tf.gfile.Exists(lang1_fname) and tf.gfile.Exists(lang2_fname) and tf.gfile.Exists(lang2a_fname):
    tf.logging.info("Skipping compile data, found files:\n%s\n%s\n%s", lang1_fname,lang2_fname,lang2a_fname)
    return filename
  with tf.gfile.GFile(lang1_fname, mode="w") as lang1_resfile:
    with tf.gfile.GFile(lang2_fname, mode="w") as lang2_resfile:
      with tf.gfile.GFile(lang2a_fname, mode="w") as lang2a_resfile:
        for dataset in datasets:
          url = dataset[0]
          compressed_filename = os.path.basename(url)
          compressed_filepath = os.path.join(tmp_dir, compressed_filename)
          if url.startswith("http"):
            generator_utils.maybe_download(tmp_dir, compressed_filename, url)
          if compressed_filename.endswith(".zip"):
            zipfile.ZipFile(os.path.join(compressed_filepath),"r").extractall(tmp_dir)

          if dataset[1][0] == "tmx":
            raise NotImplementedError()
          elif dataset[1][0] == "tsv":
            raise NotImplementedError()
          else:
            lang1_filename, lang2_filename, lang2a_filename = dataset[1]
            lang1_filepath = os.path.join(tmp_dir, lang1_filename)
            lang2_filepath = os.path.join(tmp_dir, lang2_filename)
            lang2a_filepath = os.path.join(tmp_dir, lang2a_filename)
            is_sgm = (
              lang1_filename.endswith("sgm") and lang2_filename.endswith("sgm") and lang2a_filename.endswith("sgm"))
            if not (tf.gfile.Exists(lang1_filepath) and tf.gfile.Exists(lang2_filepath) and tf.gfile.Exists(lang2a_filepath)):
              # For .tar.gz and .tgz files, we read compressed.
              mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
              with tarfile.open(compressed_filepath, mode) as corpus_tar:
                corpus_tar.extractall(tmp_dir)
            if lang1_filepath.endswith(".gz"):
              new_filepath = lang1_filepath.strip(".gz")
              generator_utils.gunzip_file(lang1_filepath, new_filepath)
              lang1_filepath = new_filepath
            if lang2_filepath.endswith(".gz"):
              new_filepath = lang2_filepath.strip(".gz")
              generator_utils.gunzip_file(lang2_filepath, new_filepath)
              lang2_filepath = new_filepath
            if lang2a_filepath.endswith(".gz"):
              new_filepath = lang2a_filepath.strip(".gz")
              generator_utils.gunzip_file(lang2a_filepath, new_filepath)
              lang2a_filepath = new_filepath

            for example in wmt_align_iterator(lang1_filepath, lang2_filepath, lang2a_filepath):
              line1res = translate._preprocess_sgm(example["inputs"], is_sgm)
              line2res = translate._preprocess_sgm(example["ground_truth"], is_sgm)
              line3res = translate._preprocess_sgm(example["targets"], is_sgm)
              # we do not clean again since data is compiled once
              lang1_resfile.write(line1res)
              lang1_resfile.write("\n")
              lang2_resfile.write(line2res)
              lang2_resfile.write("\n")
              lang2a_resfile.write(line3res)
              lang2a_resfile.write("\n")              

  return filename

def wmt_align_iterator(input_txt_path,ground_truth_txt_path,target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  log_info("running align iterator...")
  for inputs, ground_truth, targets in zip(
      text_problems.txt_line_iterator(input_txt_path),text_problems.txt_line_iterator(ground_truth_txt_path), text_problems.txt_line_iterator(target_txt_path)):
    yield {"inputs": inputs, "ground_truth": ground_truth, "targets": targets}

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

_DEEN_ALIGN_TRAIN_DATASETS=[
    [
        "",
        ("wmt_align/wmt-deen.train.lang1", "wmt_align/wmt-deen.train.lang2","wmt_align/wmt-deen.train.lang2a")
    ],
]

_DEEN_ALIGN_EVAL_DATASETS=[
    [
        "",
        ("wmt_align/wmt-deen.dev.lang1", "wmt_align/wmt-deen.dev.lang2","wmt_align/wmt-deen.dev.lang2a")
    ],
]

_DEEN_ALIGN_TEST_DATASETS=[
    [
        "",
        ("wmt_align/wmt-deen.test.lang1", "wmt_align/wmt-deen.test.lang2","wmt_align/wmt-deen.test.lang2a")
    ],
]

_JAEN_ALIGN_TRAIN_DATASETS=[
    [
        "",
        ("wmt_align/wmt-jaen.train.lang1", "wmt_align/wmt-jaen.train.lang2","wmt_align/wmt-jaen.train.lang2a")
    ],
]

_JAEN_ALIGN_EVAL_DATASETS=[
    [
        "",
        ("wmt_align/wmt-jaen.dev.lang1", "wmt_align/wmt-jaen.dev.lang2","wmt_align/wmt-jaen.dev.lang2a")
    ],
]

_JAEN_ALIGN_TEST_DATASETS=[
    [
        "",
        ("wmt_align/wmt-jaen.test.lang1", "wmt_align/wmt-jaen.test.lang2","wmt_align/wmt-jaen.test.lang2a")
    ],
]

@registry.register_problem
class WmtDeenAlign32k(WmtDeenReorder32k):
  def was_reversed(self):
    if self._was_reversed is False:
      raise NotImplementedError('Not support reverse operation')
    return self_was_reversed

  #def use_vocab_from_other_problem(self):
  #  return WmtReorder32k()

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (super(WmtDeenAlign32k,
                                                 self)
                                           .example_reading_spec())
    data_fields["ground_truth"] = tf.VarLenFeature(tf.int64)
    return (data_fields, data_items_to_decoders)

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
    encoders = {"targets": encoder, "ground_truth": encoder}
    if self.has_inputs:
      encoders["inputs"] = encoder
    return encoders

  def source_data_files(self, dataset_split):
    train_datasets = _DEEN_ALIGN_TRAIN_DATASETS + self.additional_training_datasets
    if dataset_split == problem.DatasetSplit.TRAIN:
      return train_datasets
    elif dataset_split == problem.DatasetSplit.EVAL:
      return _DEEN_ALIGN_EVAL_DATASETS
    else:
      return _DEEN_ALIGN_TEST_DATASETS  

  def generate_samples(self,
                       data_dir,
                       tmp_dir,
                       dataset_split,
                       custom_iterator=wmt_align_iterator):
    datasets = self.source_data_files(dataset_split)
    tag = "test"
    datatypes_to_clean = None
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = "train"
      datatypes_to_clean = self.datatypes_to_clean
    elif dataset_split == problem.DatasetSplit.EVAL:
      tag = "dev"
    data_path = compile_aligned_data(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag),
        datatypes_to_clean=datatypes_to_clean)

    return custom_iterator(data_path + ".lang1", data_path + ".lang2", data_path + ".lang2a")

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

@registry.register_problem
class WmtJaenAlign32k(WmtDeenAlign32k):
  def source_data_files(self, dataset_split):
    train_datasets = _JAEN_ALIGN_TRAIN_DATASETS + self.additional_training_datasets
    if dataset_split == problem.DatasetSplit.TRAIN:
      return train_datasets
    elif dataset_split == problem.DatasetSplit.EVAL:
      return _JAEN_ALIGN_EVAL_DATASETS
    else:
      return _JAEN_ALIGN_TEST_DATASETS      

from tensor2tensor.utils import data_reader
import functools

def reorder_input_fn(dataset,
             filepattern,
             skip_random_fraction_when_training,
             batch_size_means_tokens_param,
             batch_size_multiplier,
             max_length,
             mode,
             hparams,
             data_dir=None,
             params=None,
             config=None,
             force_repeat=False,
             prevent_repeat=False):
  """Builds input pipeline for problem. Completely identical with original
     apart from one line in prepare_for_output(example) function

  Args:
    dataset: the dataset to make input function from.
    filepattern: the pattern of files to read from.
    skip_random_fraction_when_training: whether to skip randomly when training.
    batch_size_means_tokens_param: whether batch size should mean tokens.
    batch_size_multiplier: how to multiply batch size when bucketing.
    max_length: maximum length,
    mode: tf.estimator.ModeKeys
    hparams: HParams, model hparams
    data_dir: str, data directory; if None, will use hparams.data_dir
    params: dict, may include "batch_size"
    config: RunConfig; should have the data_parallelism attribute if not using
      TPU
    force_repeat: bool, whether to repeat the data even if not training
    prevent_repeat: bool, whether to not repeat when in training mode.
      Overrides force_repeat.

  Returns:
    (features_dict<str name, Tensor feature>, Tensor targets)
  """
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  if config and config.use_tpu:
    num_threads = 64
  else:
    num_threads = data_reader.cpu_count() if is_training else 1

  if config and hasattr(config,
                        "data_parallelism") and config.data_parallelism:
    num_shards = config.data_parallelism.n
  else:
    num_shards = 1

  mlperf_log.transformer_print(
      key=mlperf_log.INPUT_MAX_LENGTH, value=max_length)

  def tpu_valid_size(example):
    return data_reader.example_valid_size(example, hparams.min_length, max_length)

  def gpu_valid_size(example):
    drop_long_sequences = is_training or hparams.eval_drop_long_sequences
    max_validate_length = max_length if drop_long_sequences else 10**9
    return data_reader.example_valid_size(example, hparams.min_length, max_validate_length)

  def define_shapes(example):
    batch_size = config and config.use_tpu and params["batch_size"]
    return data_reader.standardize_shapes(example, batch_size=batch_size)

  # Read and preprocess
  data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)

  if (force_repeat or is_training) and not prevent_repeat:
    # Repeat and skip a random number of records
    dataset = dataset.repeat()

  if is_training and skip_random_fraction_when_training:
    data_files = contrib.slim().parallel_reader.get_data_files(filepattern)
    #  In continuous_train_and_eval when switching between train and
    #  eval, this input_fn method gets called multiple times and it
    #  would give you the exact same samples from the last call
    #  (because the Graph seed is set). So this skip gives you some
    #  shuffling.
    dataset = data_reader.skip_random_fraction(dataset, data_files[0])

  dataset = dataset.map(data_reader.cast_ints_to_int32, num_parallel_calls=num_threads)

  if batch_size_means_tokens_param:
    batch_size_means_tokens = True
  else:
    if _are_shapes_fully_defined(dataset.output_shapes):
      batch_size_means_tokens = False
    else:
      tf.logging.warning(
          "Shapes are not fully defined. Assuming batch_size means tokens.")
      batch_size_means_tokens = True

  # Batching
  if not batch_size_means_tokens:
    # Batch size means examples per datashard.
    if config and config.use_tpu:
      # on TPU, we use params["batch_size"], which specifies the number of
      # examples across all datashards
      batch_size = params["batch_size"]
      dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
      batch_size = hparams.batch_size * num_shards
      dataset = dataset.batch(batch_size)
  else:
    # batch_size means tokens per datashard
    if config and config.use_tpu:
      dataset = dataset.filter(tpu_valid_size)
      padded_shapes = pad_for_tpu(dataset.output_shapes, hparams, max_length)
      # on TPU, we use params["batch_size"], which specifies the number of
      # examples across all datashards
      batch_size = params["batch_size"]
      if hparams.pad_batch:
        tf.logging.warn(
            "Padding the batch to ensure that remainder eval batches are "
            "processed. This may lead to incorrect metrics for "
            "non-zero-padded features, e.g. images. Use a smaller batch "
            "size that has no remainder in that case.")
        dataset = dataset.padded_batch(
            batch_size, padded_shapes, drop_remainder=False)
        dataset = dataset.map(
            functools.partial(data_reader.pad_batch, batch_multiple=batch_size),
            num_parallel_calls=num_threads)
      else:
        dataset = dataset.padded_batch(
            batch_size, padded_shapes, drop_remainder=True)
    else:
      # On GPU, bucket by length
      dataset = dataset.filter(gpu_valid_size)
      cur_batching_scheme = data_reader.hparams_to_batching_scheme(
          hparams,
          shard_multiplier=num_shards,
          length_multiplier=batch_size_multiplier)
      if hparams.use_fixed_batch_size:
        # Here  batch_size really means examples per datashard.
        cur_batching_scheme["batch_sizes"] = [hparams.batch_size]
        cur_batching_scheme["boundaries"] = []
      dataset = dataset.apply(
          tf.data.experimental.bucket_by_sequence_length(
              data_reader.example_length, cur_batching_scheme["boundaries"],
              cur_batching_scheme["batch_sizes"]))

      if not is_training:
        batch_multiple = num_shards
        if hparams.use_fixed_batch_size:
          # Make sure the last batch has the same fixed size as the rest.
          batch_multiple *= hparams.batch_size
        if batch_multiple > 1:
          tf.logging.warn(
              "Padding the batch to ensure that remainder eval batches have "
              "a batch size divisible by the number of data shards. This may "
              "lead to incorrect metrics for non-zero-padded features, e.g. "
              "images. Use a single datashard (i.e. 1 GPU) in that case.")
          dataset = dataset.map(
              functools.partial(data_reader.pad_batch, batch_multiple=batch_multiple),
              num_parallel_calls=num_threads)

  dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)

  # Add shuffling for training batches. This is necessary along with record
  # level shuffling in the dataset generation. Record shuffling will shuffle
  # the examples. However, in some cases, it's possible that the shuffle
  # buffer size for record shuffling is smaller than the batch size. In such
  # cases, adding batch shuffling ensures that the data is in random order
  # during training
  if (is_training and hasattr(hparams, "batch_shuffle_size") and
      hparams.batch_shuffle_size):
    dataset = dataset.shuffle(hparams.batch_shuffle_size)

  # Split batches into chunks if targets are too long.
  # The new "chunk_number" feature is 0 for the first chunk and goes up then.
  # Chunks are reversed so the 0th chunk comes first, then the 1st and so on,
  # so models can attend to them in the order they arrive. The last chunk is
  # usually the one containing the end of the target sentence (EOS).
  chunk_length = hparams.get("split_targets_chunk_length", 0)
  max_chunks = hparams.get("split_targets_max_chunks", 100)
  if chunk_length > 0:
    def is_nonzero_chunk(example):
      """A chunk is zero if all targets are 0s."""
      return tf.less(0, tf.reduce_sum(tf.abs(example["targets"])))

    def split_on_length(example):
      """Split a batch of ditcs on length."""
      x = example["targets"]
      # TODO(kitaev): This code breaks if chunk_length * max_chunks < batch_size
      length_diff = chunk_length * max_chunks - tf.shape(x)[1]
      padded_x = tf.pad(x, [(0, 0), (0, length_diff), (0, 0), (0, 0)])
      chunks = [padded_x[:, i*chunk_length:(i+1)*chunk_length, :, :]
                for i in range(max_chunks - 1)]
      chunks.append(padded_x[:, (max_chunks - 1)*chunk_length:, :, :])
      new_example = {}
      # Setting chunk_number to be tf.range(max_chunks) is incompatible with TPU
      new_example["chunk_number"] = tf.concat([
          tf.expand_dims(tf.ones_like(c) * n, axis=0)
          for n, c in enumerate(chunks)
      ],
                                              axis=0)
      new_example["targets"] = tf.concat(
          [tf.expand_dims(c, axis=0) for c in chunks], axis=0)
      for k in example:
        if k != "targets":
          assert k != "chunk_number", (
              "Chunking code expects the chunk_number feature name to be "
              "available"
          )
          new_example[k] = tf.concat(
              [tf.expand_dims(example[k], axis=0) for _ in range(max_chunks)],
              axis=0)
      return tf.data.Dataset.from_tensor_slices(new_example)

    dataset = dataset.flat_map(split_on_length)
    dataset = dataset.filter(is_nonzero_chunk)

    # The chunking data pipeline thus far creates batches of examples where all
    # of the examples have the same chunk number. This can lead to periodic
    # fluctuations in the loss; for example, when all examples in the batch have
    # chunk number 0 the loss may be higher than midway through a sequence.
    # Enabling split_targets_strided_training adjusts the data so that each
    # batch includes examples at various points within a sequence.
    if is_training and hparams.split_targets_strided_training:
      # TODO(kitaev): make sure that shape inference works on GPU, not just TPU.
      inferred_batch_size = dataset.output_shapes["targets"].as_list()[0]
      if inferred_batch_size is None:
        raise ValueError(
            "Strided training is only implemented when the batch size can be "
            "inferred statically, for example when training on TPU."
        )
      chunk_stride = inferred_batch_size * max(
          1, max_chunks // inferred_batch_size) + 1

      def collapse_nested_datasets(example):
        """Converts a dataset of datasets to a dataset of tensor features."""
        new_example = {}
        for k, v in example.items():
          v = tf.data.experimental.get_single_element(
              v.batch(inferred_batch_size, drop_remainder=True))
          new_example[k] = v
        return tf.data.Dataset.from_tensor_slices(new_example)

      dataset = dataset.unbatch()
      dataset = dataset.window(inferred_batch_size, inferred_batch_size,
                               chunk_stride)
      dataset = dataset.flat_map(collapse_nested_datasets)
      dataset = dataset.batch(inferred_batch_size, drop_remainder=True)

  def prepare_for_output(example):
    if not config or not config.use_tpu:
      data_reader._summarize_features(example, num_shards)
    # the only difference is here instead of example.pop('targets')
    if mode == tf.estimator.ModeKeys.PREDICT:
      example["infer_targets"] = example["targets"] 
      return example
    else:
      return example, example[hparams.get(
          key="labels_feature_name", default="targets")]

  dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
  dataset = dataset.prefetch(2)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # This is because of a bug in the Estimator that short-circuits prediction
    # if it doesn't see a QueueRunner. DummyQueueRunner implements the
    # minimal expected interface but does nothing.
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_reader.DummyQueueRunner())

  return dataset

def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True,
                               inputs_prefix="",
                               targets_prefix=""):
  """Encode Text2Text samples from the generator with the vocab."""
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    src_sent = vocab.encode(inputs_prefix + sample["inputs"])
    trg_sent = targets_vocab.encode(targets_prefix + sample["targets"])
    src_sent.append(text_encoder.EOS_ID)
    #our model does not need to learn end of sentence for the output
    #trg_sent.append(text_encoder.EOS_ID)
    sample["inputs"]=src_sent
    sample["targets"]=trg_sent
    yield sample
