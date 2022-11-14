from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import transformer_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import beam_search
from tensor2tensor.utils.t2t_model import log_info

from scipy.optimize import linear_sum_assignment

import tensorflow as tf
import six

"""function utils"""

def aligner_dot(input, to_dot, nonpadding=None):
  if nonpadding is None:
    padding=common_attention.embedding_to_padding(to_dot)
  else: 
    padding=tf.reshape(tf.squeeze(1-nonpadding),tf.shape(nonpadding)[:2])
  length = common_attention.padding_to_length(padding)
  i_shape=common_layers.shape_list(input)
  d_shape=common_layers.shape_list(to_dot)
  with tf.variable_scope("dot_product_attention", values=[input, to_dot]) as scope:
    A_logits=tf.matmul(input,to_dot,transpose_b=True)
    A_logits=tf.reshape(A_logits,i_shape[:2]+[1,1,d_shape[1]])
    a_mask=tf.tile(tf.reshape(tf.cast(tf.less(
           tf.tile(tf.expand_dims(tf.range(d_shape[1]),0),[i_shape[0],1]),
           tf.tile(tf.expand_dims(length,1),[1,d_shape[1]])),
           dtype=A_logits.dtype),[i_shape[0],1,1,1,d_shape[1]]),
           [1,i_shape[1],1,1,1])
    A_logits+=tf.cast(1-a_mask,dtype=A_logits.dtype)*common_attention.large_compatible_negative(A_logits.dtype)
    A_logits = tf.reshape(A_logits,i_shape[:2]+[d_shape[1]])
    return A_logits, a_mask

def batch_upscale(align_matrix):
  #align_matrix: B, T, T
  amat = align_matrix
  a_shape=tf.shape(amat)
  length = 2*a_shape[1]-1
  tmp=tf.tile(tf.expand_dims(tf.range(a_shape[1]),1),[1,length])
  tmp=tf.tile(tf.expand_dims(tmp,1),[1,a_shape[0],1])
  def single_slice(i):
    rid=tf.stack([tf.ones(a_shape[1]-i,dtype=amat.dtype)*i,a_shape[1]-tf.range(a_shape[1]-i)-1],1)
    lid=tf.stack([a_shape[1]-tf.range(a_shape[1]-i-1)-1,tf.ones(a_shape[1]-i-1,dtype=amat.dtype)*i],1)
    rid=tf.concat([tf.tile(tf.reshape(tf.range(a_shape[0]),[-1,1,1]),[1,a_shape[1]-i,1]),tf.tile(tf.expand_dims(rid,0),[a_shape[0],1,1])],-1)
    lid=tf.concat([tf.tile(tf.reshape(tf.range(a_shape[0]),[-1,1,1]),[1,a_shape[1]-i-1,1]),tf.tile(tf.expand_dims(lid,0),[a_shape[0],1,1])],-1)
    b= tf.concat([tf.gather_nd(amat,rid),tf.reverse(tf.gather_nd(amat,lid),[-1])],1)
    padding=tf.slice(tf.zeros(length,dtype=b.dtype),[0],[(length-tf.shape(b)[1])//2])
    padding = tf.tile(tf.expand_dims(padding,0),[a_shape[0],1])
    return tf.concat([padding,b,padding],1)

  amat = tf.map_fn(single_slice,tf.range(a_shape[1]),dtype=(amat.dtype))
  
  def upscale_cond(t, i, align_matrix):
    return tf.greater(tf.shape(align_matrix)[0],i)

  def upscale_body(t, i, align_matrix):
    nt=tf.gather(align_matrix,i) #if align_matrix[i]==1 mask
    t=(t+1)*nt
    nt = tf.cond(tf.less(i+1,tf.shape(align_matrix)[0]),lambda: nt*(1-tf.gather(align_matrix,i+1)),lambda: nt) #if align_matrix[i+1]==0 mask
    cond=tf.greater(tf.reduce_sum(nt,1),0)
    def fn(t,i,align_matrix):
      mask=tf.tile(tf.expand_dims(nt,0),[a_shape[1],1,1])
      mask*=tf.cast(tf.logical_and(tf.greater(tmp,tf.ones_like(tmp)*i-t),tf.less_equal(tmp,i)),dtype=mask.dtype)
      update = tf.tile(tf.expand_dims(tf.reduce_sum(mask,0),0),[a_shape[1],1,1])
      return align_matrix*update*mask+(1-mask)*align_matrix
    align_matrix=tf.cond(tf.reduce_any(cond),lambda: fn(t,i,align_matrix),lambda: align_matrix)
    return t, i+1, align_matrix

  t,i,align_matrix=tf.while_loop(upscale_cond,upscale_body,[tf.zeros(tf.stack([a_shape[0],length]),dtype=amat.dtype),tf.constant(0),amat])

  def reverse_single_slice(i):
    row = tf.gather(align_matrix,i)
    right = tf.reverse(tf.slice(row,[0,i],[a_shape[0],a_shape[1]-i]),[1])
    left_id = tf.stack([tf.range(i),tf.reverse(a_shape[1]+tf.range(i),[0])],1)
    left_id = tf.tile(tf.expand_dims(left_id,0),[a_shape[0],1,1])
    left_id = tf.concat([left_id[:,:,:1],tf.tile(tf.reshape(tf.range(a_shape[0]),[a_shape[0],1,1]),[1,i,1]),left_id[:,:,1:]],-1)
    return tf.concat([tf.gather_nd(align_matrix,left_id),right],1)

  align_matrix = tf.map_fn(reverse_single_slice,tf.range(a_shape[1]),dtype=(align_matrix.dtype))
  
  align_matrix = tf.reshape(align_matrix,[-1,a_shape[1]])
  id=tf.reshape(tf.tile(tf.expand_dims(tf.range(a_shape[0]),1),[1,a_shape[1]]),[-1])
  id+=tf.tile(tf.range(a_shape[1]),[a_shape[0]])*a_shape[0]
  align_matrix=tf.reshape(tf.gather(align_matrix,id),a_shape)
  return align_matrix

def compute_alignment_matrix(inputs, targets, nonpadding):
  log_info('computing alignment matrix')
  t_shape = common_layers.shape_list(nonpadding)
  inputs = tf.reshape(inputs,t_shape[:2])
  targets = tf.reshape(targets,t_shape[:2])
  inputs = tf.expand_dims(inputs,-1)
  targets = tf.expand_dims(targets,1)
  inputs = tf.tile(inputs,[1,1,t_shape[1]])
  targets = tf.tile(targets,[1,t_shape[1],1])
  matrix = tf.cast(tf.equal(inputs,targets),dtype=nonpadding.dtype)
  num_equal=tf.tile(tf.expand_dims(tf.reduce_sum(matrix,-1),-1),
                    [1,1,t_shape[1]])
  # sometimes ground truth and targets might not be completely identical 
  # because of subword tokenization problem, we clip min to 1 to avoid 
  # division by 0
  matrix/=tf.clip_by_value(num_equal,1,tf.reduce_max(num_equal))
  return matrix*tf.expand_dims(nonpadding,1)

def get_bos(inputs):
  """ 
  args:
    inputs: a Tensor with shape [B,T,D] or [B,T,1,D]
  returns: 
    output: a Tensor with shape [B,1,D]
  """
  input_shape=common_layers.shape_list(inputs)
  if len(input_shape)==4:
    inputs = common_layers.flatten4d3d(inputs)
    input_shape=common_layers.shape_list(inputs)
  assert len(input_shape)==3
  return common_layers.shift_right_3d(inputs, None)[:,:1]

def lm_prepare_decoder(targets, hparams, features=None, pad=None):
  # we remove pos embedding in standard prepare_decoder and leave its
  # inclusion at the self.decode part
  # we also edited the shift_right_3d because target dont end with <eos>
  # this is only used in training
  if hparams.causal_decoder_self_attention:
    # Causal attention.
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # Full attention.
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  
  decoder_input = common_layers.shift_right_3d(targets, pad)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)  

def nalm_pos_prepare_decoder(targets, hparams, features=None, pad=None):
  # basically nar_prepare_decoder with position information
  target_space = features.get("target_space_id",0) if hparams.has_input else None
  targets, bias = transformer_layers.transformer_prepare_encoder(
                 targets,target_space,hparams,features=features)[:2]
  bias = tf.concat([bias[:,:,:,1:2],bias[:,:,:,1:]],-1)
  return targets, bias

def nalm_prepare_decoder(targets, hparams, features=None, pad=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    pad: vector to use for padding when shifting targets right

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  hparams.pos='' 
  return nalm_pos_prepare_decoder(targets, hparams, features=features, pad=pad)

def position_attention_layer(input, bias, layer_idx, hparams):
  """position attention layer which is found in snat-r"""
  layer = layer_idx
  layer_name = "layer_%d" % layer
  x_shape = common_layers.shape_list(input)
  num_heads = hparams.num_heads
  signal = common_attention.get_timing_signal_1d(x_shape[1], x_shape[2], 1.0, 1.0e4,0)
  signal = common_layers.cast_like(signal,input)
  with tf.variable_scope(layer_name):
    with tf.variable_scope("position_attention"):
      x = common_layers.layer_preprocess(input, hparams)
      q,_,v=common_attention.compute_qkv(signal, x, 
                                         hparams.hidden_size, 
                                         hparams.hidden_size)
      q = common_attention.split_heads(q, num_heads)
      v = common_attention.split_heads(v, num_heads)
      logits = tf.matmul(q, q, transpose_b=True)
      if bias is not None:
        bias = common_layers.cast_like(bias, logits)
        logits += bias
      weights = tf.nn.softmax(logits)
      weights = common_layers.cast_like(weights, q)
      output = tf.matmul(weights,v)
  return common_attention.combine_heads(output)

def regularized_decoder_layer(
          decoder_input,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=None,
          encoder_output=None,
          cache=None,
          decode_loop_step=None,
          nonpadding=None,
          save_weights_to=None,
          make_image_summary=False,
          losses=None,
          layer_collection=None,
          recurrent_memory_by_layer=None,
          chunk_number=None):
  """Transformer decoder layer with added position attention layer"""
  x, layer_cache = transformer.transformer_self_attention_layer(
      decoder_input=decoder_input,
      decoder_self_attention_bias=decoder_self_attention_bias,
      layer_idx=layer_idx,
      hparams=hparams,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      cache=cache,
      decode_loop_step=decode_loop_step,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      layer_collection=layer_collection,
      recurrent_memory_by_layer=recurrent_memory_by_layer,
      chunk_number=chunk_number)

  y = position_attention_layer(x, decoder_self_attention_bias, layer_idx, hparams)
  x = common_layers.layer_postprocess(x, y, hparams)

  layer = layer_idx
  layer_name = "layer_%d" % layer
  with tf.variable_scope(layer_name):
    with tf.variable_scope("ffn"):
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(
              x, hparams, layer_collection=layer_collection),
          hparams,
          conv_padding="LEFT",
          nonpadding_mask=nonpadding,
          losses=losses,
          cache=layer_cache,
          decode_loop_step=decode_loop_step,
          layer_collection=layer_collection)
      x = common_layers.layer_postprocess(x, y, hparams)
      return x

def regularized_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None,
                        layer_output_cache=None):
  """A stack of transformer layers with position attention layer.
     The rest follow the transformer_decoder function
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  final_layer = hparams.num_decoder_layers or hparams.num_hidden_layers -1
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)

  with tf.variable_scope(name):
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x = regularized_decoder_layer(
          x,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          nonpadding=nonpadding,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          losses=losses,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number
          )
      if final_layer>layer_idx and layer_idx>0 and layer_output_cache is not None:
        with tf.variable_scope("layer_%d" % final_layer, reuse=tf.AUTO_REUSE):
          with tf.variable_scope("ffn"):
            tmp_y = transformer.transformer_ffn_layer(
              common_layers.layer_preprocess(
              x, hparams, layer_collection=layer_collection),
              hparams,
              conv_padding="LEFT",
              nonpadding_mask=nonpadding)
            tmp_x = common_layers.layer_postprocess(x, tmp_y, hparams)
        layer_output_cache["layer_%d_output" % layer_idx] = tf.expand_dims(x, axis=2)

    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams, layer_collection=layer_collection)

def select_symbol_top(idx, model_hparams, vocab_size):
  """select custom vectors from top vocab.

  Args:
    idx: A Tensor with shape [batch, 1].
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    selected_vocab: A Tensor with shape  [batch, hidden_size].
  """
  if model_hparams.shared_embedding_and_softmax_weights:
    scope_name = "shared"
    reuse = tf.AUTO_REUSE
  else:
    scope_name = "softmax"
    reuse = False
  with tf.variable_scope(scope_name, reuse=reuse):
    input_shape = common_layers.shape_list(idx)
    var = modalities.get_weights(model_hparams, vocab_size)
    return tf.gather(var,idx)

def permutation_weights(inputs, nonpadding, hparams):
  num_heads = 1
  layer_size = hparams.num_encoder_layers
  hparams.num_encoder_layers = 1
  inputs = common_layers.flatten4d3d(inputs)
  bias = common_attention.attention_bias_ignore_padding(1.0-nonpadding)
  x = transformer_layers.transformer_encoder(inputs, bias, hparams, name='pre_network')
  x1 = transformer_layers.transformer_encoder(x, bias, hparams, name='q_network')
  x2 = transformer_layers.transformer_encoder(x, bias, hparams, name='k_network')
  hparams.num_encoder_layers = layer_size
  with tf.variable_scope("permutation_matrix",values=[x1,x2]):
    q,k,_=common_attention.compute_qkv(x1,x2, hparams.hidden_size, hparams.hidden_size)
    g = tf.sigmoid(tf.matmul(q,tf.cast(tf.get_variable("u",[hparams.hidden_size,1]),dtype=q.dtype)))
    g = tf.reshape(g, common_layers.shape_list(g)[:-1])
    diagonal = tf.linalg.diag(tf.ones_like(nonpadding)) #diagonal with 1 as value
    #dg = diagonal * g
    pred, mask = aligner_dot(q,k,nonpadding)
    pred += tf.linalg.diag(nonpadding)*common_attention.large_compatible_negative(q.dtype) # M
    pred = tf.clip_by_value(pred,common_attention.large_compatible_negative(q.dtype),tf.reduce_max(pred))
    return diagonal * tf.linalg.diag(g) + (1-diagonal) * tf.tile(tf.expand_dims(
	(1-g),-1),[1,1,common_layers.shape_list(g)[-1]]) * tf.nn.softmax(pred), mask

def klx(x, y):
  def fixprob(att):
    att = att + 1e-9
    _sum = tf.reduce_sum(att, reduction_indices=1, keep_dims=True)
    att = att / _sum
    att = tf.clip_by_value(att, 1e-9, 1.0, name=None)
    return att

  x = fixprob(x)
  y = fixprob(y)
  X = tf.distributions.Categorical(probs=x)
  Y = tf.distributions.Categorical(probs=y)
  return tf.distributions.kl_divergence(X, Y)

def compute_alignment_matrix(inputs, targets, nonpadding):
  log_info('computing alignment matrix')
  t_shape = common_layers.shape_list(nonpadding)
  inputs = tf.reshape(inputs,t_shape[:2])
  targets = tf.reshape(targets,t_shape[:2])
  inputs = tf.expand_dims(inputs,-1)
  targets = tf.expand_dims(targets,1)
  inputs = tf.tile(inputs,[1,1,t_shape[1]])
  targets = tf.tile(targets,[1,t_shape[1],1])
  matrix = tf.cast(tf.equal(inputs,targets),dtype=nonpadding.dtype)
  num_equal=tf.tile(tf.expand_dims(tf.reduce_sum(matrix,-1),-1),[1,1,t_shape[1]])
  # sometimes ground truth and targets might not be completely identical because of subword tokenization problem, we clip min to 1 to avoid division by 0
  matrix/=tf.clip_by_value(num_equal,1,tf.reduce_max(num_equal))
  return matrix*tf.expand_dims(nonpadding,1)

"""decoding algo utils"""

def combine_viterbi(prob_M, prob_R, mask=None, mask_window=1):
  #consider more than 1 seq [Batch, length, length]
  #mask_window=window_size
  if mask is None:
    mask = tf.cast(tf.greater(prob_M,0),dtype=prob_M.dtype)[:,1:,1:]
  else:
    mask = tf.tile(tf.expand_dims(tf.cast(mask[:,1:],dtype=prob_M.dtype),1),
	       [1,tf.shape(mask)[-1]-1,1])

  len_ids = tf.cast(tf.reduce_sum(mask[:,0],1),dtype=tf.int32)-1
  #prob_M = tf.log(prob_R + 1e-37) #turn into log prob, 1e-38 = 0
  #prob_M = tf.transpose(tf.log(prob_R + 1e-37),[0,2,1])
  prob_M = tf.log(prob_M + 1e-37) 
  if prob_R is not None:
    prob_M += tf.log(prob_R + 1e-37)
  prob_M = tf.clip_by_value(prob_M,
           common_attention.large_compatible_negative(prob_M.dtype),0)
  trellis, score = tf.expand_dims(prob_M[:,0,1:],1), prob_M[:,1:,1:]
  batch_size, seq_len, _ = common_layers.shape_list(score) #seq_len = num_tag
  if mask_window>1:
    m = common_layers.ones_matrix_band_part(seq_len,
                                            seq_len,
                                            mask_window,
                                            mask_window,
                                            out_shape=[1,seq_len,seq_len])
    score = score*m + (1-m)*tf.log(1e-37)/2
  base_index=tf.range(seq_len)
  backpointers=tf.tile(tf.reshape(base_index,[1,1,-1]),[batch_size,1,1])
  mask *= 1.0-tf.tile(tf.cast(tf.expand_dims(tf.scatter_nd(
              tf.stack([base_index,base_index],-1), 
              tf.ones_like(tf.range(seq_len),dtype=prob_M.dtype),
              [seq_len,seq_len]),0),prob_M.dtype),[batch_size,1,1])
  mask = 1.0 - mask

  def end_cond(m, t, b):
    return tf.not_equal(tf.reduce_sum(1-m),0)

  def while_body(mask, trellis, backpointers):
    old_mask = mask
    mask*=common_attention.large_compatible_negative(mask.dtype)
    v = tf.expand_dims(trellis[:,-1], -1) + tf.gather_nd(score,
          tf.expand_dims(backpointers[:,-1],-1),batch_dims=1) + mask
    val,ind=tf.nn.top_k(v,1)
    trellis = tf.concat([trellis,tf.transpose(val,[0,2,1])],1)
    trellis = tf.clip_by_value(trellis,
                common_attention.large_compatible_negative(mask.dtype),0.0) # clip so that value wont overflow
    backpointers = tf.concat([backpointers,tf.transpose(ind,[0,2,1])],1)
    round = common_layers.shape_list(backpointers)[1]
    g = tf.reshape(tf.stack([tf.tile(tf.reshape(base_index,[1,-1,1]),
        [batch_size,1,round]),tf.transpose(backpointers,[0,2,1])],-1),[-1,2])
    g = tf.concat([tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size),1),
          [1,seq_len*round]),[-1,1]),g],-1) #adding batch index
    mask = tf.clip_by_value(old_mask+tf.scatter_nd(g, 
             tf.cast(tf.ones_like(tf.range(batch_size*round*seq_len)),
             dtype=mask.dtype),[batch_size,seq_len,seq_len]),0,1)
    return mask, trellis, backpointers

  m,t,b = tf.while_loop(end_cond, while_body,[mask, trellis, backpointers],
                         shape_invariants=[tf.TensorShape([None,None,None]),
                                           tf.TensorShape([None,None,None]),
                                           tf.TensorShape([None,None,None])])

  nt = tf.gather_nd(t,tf.expand_dims(len_ids,-1),batch_dims=1)+tf.gather_nd(
         prob_M[:,1:,0],tf.expand_dims(tf.gather_nd(b,tf.expand_dims(len_ids,
         -1),batch_dims=1),-1),batch_dims=1)
  t = tf.concat([t,tf.expand_dims(nt,1)],1)
  order = tf.gather_nd(tf.transpose(b,[0,2,1]),
            tf.stack([tf.range(batch_size),
            tf.argmax(nt,1,output_type=tf.int32)],-1))
  return order,m,t,b

def combine_align(output, A_matrix, B_matrix, f_mask, mask_window=1):
  # mask_window is instead window size
  # forward and backward matrices
  # with dimension <bos>+L * <eos>+L
  order,mask,_,_ = combine_viterbi(A_matrix,B_matrix,
    mask=f_mask,mask_window=mask_window)
  #shape is dynamic, use old gather method
  x_shape=common_layers.shape_list(output)
  order+=tf.tile(tf.expand_dims(tf.range(x_shape[0]),1),
           [1,x_shape[1]])*x_shape[1]
  out_seq = tf.reshape(tf.gather_nd(tf.reshape(output,[-1]),
              tf.reshape(order,[-1,1])),x_shape)
  f_mask=f_mask[:,1:]
  return out_seq*f_mask + beam_search.EOS_ID*(1-f_mask)

@tf.function
def tf_linear_sum_assignment(cost_matrix):
  return tf.numpy_function(func=linear_sum_assignment,inp=[cost_matrix],Tout=[tf.int64,tf.int64])

def dummy_wrap_scipy_hungarian(cost_matrix):
  order = tf_linear_sum_assignment(cost_matrix)[1]
  return tf.cast(order,dtype=cost_matrix.dtype)

def hungarian(prob_M, nonpadding):
  m_shape = common_layers.shape_list(prob_M)
  base_star = tf.linalg.diag(1-nonpadding)
  matrix_nonpadding = tf.tile(tf.expand_dims(tf.cast(nonpadding,dtype=prob_M.dtype),1),[1,m_shape[-1],1])*tf.tile(tf.expand_dims(tf.cast(nonpadding,dtype=prob_M.dtype),-1),[1,1,m_shape[-1]])
  init_matrix = -tf.log(prob_M*matrix_nonpadding+1e-37)
  ids = tf.map_fn(lambda x: dummy_wrap_scipy_hungarian(x),init_matrix)
  order = tf.cast(ids,nonpadding.dtype)
  star_mat = tf.one_hot(order,m_shape[-1])
  score = tf.reduce_sum(tf.reshape(init_matrix*matrix_nonpadding*tf.cast(star_mat,dtype=init_matrix.dtype),[m_shape[0],-1]),1)
  order = tf.cast(tf.tile(tf.expand_dims(tf.range(m_shape[-1]),0),[m_shape[0],1]),nonpadding.dtype)*(1-nonpadding)+order*nonpadding 
  return order, score 

def perm_align(output, A_matrix, f_mask):
    # with dimension L * L , no bos, no eos
    #A_matrix = tf.Print(A_matrix,[A_matrix[-1]],summarize=25600)
    order,score = hungarian(A_matrix,f_mask)
    #order = tf.Print(order,[order],summarize=25600, message='ORDER: ')
    x_shape=common_layers.shape_list(output)
    order+=tf.cast(tf.tile(tf.expand_dims(tf.range(x_shape[0]),1),[1,x_shape[1]])*x_shape[1],dtype=order.dtype)
    out_seq = tf.reshape(tf.gather_nd(tf.reshape(output,[-1]),tf.reshape(order,[-1,1])),x_shape)
    return out_seq*f_mask + beam_search.EOS_ID*(1-f_mask)

@registry.register_model
class lm(transformer.Transformer):
  #this is a standard transformer but adapted for lm-based linearization
  def __init__(self, *args, **kwargs):
    super(ar_base, self).__init__(*args, **kwargs)
    self._prepare_decoder_fn = lm_prepare_decoder
    self._hparams.has_input = self.has_input

  def model_fn(self, features):
    with tf.variable_scope(tf.get_variable_scope(), use_resource=True) as vs:
      self._add_variable_scope("model_fn", vs)
      features = add_eos_to_feature_target_ids(features)
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      self.decoder_input = common_layers.flatten4d3d(transformed_features['targets'])

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        body_out = self.body(transformed_features)
      output, losses = self._normalize_body_output(body_out)

      if "training" in losses:
        log_info("Skipping T2TModel top and loss because training loss "
                 "returned from body")
        logits = output
      else:
        logits = self.top(output, features)
        losses["training"] = 0.0
        if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
            self._hparams.mode != "attack"):
          losses["training"] = self.loss(logits, features)

      return logits, losses

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             mode='TRAIN',
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    #decoder_input is shifted
    if mode == 'TRAIN':
      decoder_input = tf.concat([get_bos(decoder_input),
                        decoder_input[:,1:]],1)
      if hparams.pos == "timing":
        decoder_input = common_attention.add_timing_signal_1d(decoder_input)
      elif hparams.pos == "timing_from_features":
        decoder_input = common_attention.add_timing_signals_from_features(
        decoder_input, features, hparams.position_features)
      elif hparams.pos == "emb":
        decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        None)
    else:
      self.decoder_input = cache['decoder_input']

    input_shape = common_layers.shape_list(decoder_input)

    decoder_output = transformer.transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, 
        losses=losses,**kwargs)

    if mode == 'TRAIN':
      padding = common_attention.embedding_to_padding(self.decoder_input)
      decoder_output*=tf.tile(tf.reshape(1.0-padding,input_shape[:2]+[1,1]),
                        [1,1,1,input_shape[-1]])
    return decoder_output

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0,
                   preprocess_targets_method=None):
    #need to edit the symbols_to_logits_fn to fix the top
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      preprocess_targets_method: method used to preprocess targets. If None,
      uses method "preprocess_targets" defined inside this method.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    if self.has_input:
      inputs_shape = common_layers.shape_list(features["inputs"])
      decode_length = inputs_shape[1]
      batch_size = inputs_shape[0]
      inputs = self._prepare_inputs_for_decode(features)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features.get("target_space_id",0),
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = features.get("targets")
      if partial_targets is None:
        partial_targets = features.get("partial_targets")
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      features["targets"] = features["inputs"]
      partial_targets = features["targets"]
      assert partial_targets is not None

    cache = {}
    if partial_targets is not None:
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = partial_targets_length
      batch_size = partial_targets_shape[0]
      nonpadding = tf.greater(partial_targets,0)
      partial_targets = add_eos_to_target_ids(partial_targets)
      cache["labels"] = partial_targets
      #partial_targets = tf.Print(partial_targets,[partial_targets[0]],summarize=256,message='initial_ids: ')
      # here we embed the targets and do an averaging
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        decoder_input = dp(bottom, partial_targets, hparams, target_vocab_size)[0]
        bos = dp(get_masked_token, decoder_input, hparams)
        cache['decoder_input'] = common_layers.flatten4d3d(decoder_input)#dp(self._prepare_decoder_fn,common_layers.flatten4d3d(decoder_input),hparams)[0]
        cache['bos'] = bos[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "timing_from_features":
      positional_encoding = common_attention.add_timing_signals_from_features(
          tf.zeros([1, decode_length, hparams.hidden_size]), features,
          hparams.position_features)
    elif hparams.pos == "emb":
      positional_enco.ding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
          "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i, cache):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)
      target_shape = common_layers.shape_list(targets)

      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      assert self.get_decode_start_id() is None #start id is avg
      if not self.get_decode_start_id():
        targets = tf.cond(
            tf.equal(i, 0), lambda: cache['bos'], lambda: targets)

      if positional_encoding is not None:
        targets += positional_encoding[:, i:i+1]
        #targets += tf.cond(tf.equal(i,0),lambda: 0.0, lambda: positional_encoding[:, i:i + 1])
      targets = tf.reshape(targets,target_shape)
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    # Create tensors for encoder-decoder attention history
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

    if not preprocess_targets_method:
      preprocess_targets_method = preprocess_targets

    #features = add_eos_to_target_ids(features)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      a=tf.reduce_sum(tf.one_hot(cache['labels'],target_vocab_size),1)
      b=tf.reduce_sum(tf.one_hot(ids, target_vocab_size),1)
      length=tf.reduce_sum(tf.cast(tf.greater(cache['labels'],0),dtype=tf.int32),1)
      #b=tf.Print(b,[i,length[1],tf.reduce_max((a-b)[1][2:]),tf.argmax((a-b)[1][2:]),cache["labels"][1],ids[1]],summarize=256)
      mask = tf.concat([tf.greater_equal(tf.expand_dims(length,1),i),tf.not_equal(tf.expand_dims(length-1,1),i)],1) #block pad while token not exhausted, insert eos when seq reach length, start with initial_id 0
      mask = tf.concat([mask,tf.equal(a-b,0)[:,2:]],1)#find all invalid tokens
      targets = tf.expand_dims(tf.expand_dims(ids[:, -1:], axis=2), axis=3)
      targets = preprocess_targets_method(targets, i, cache)
      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            mode = 'INFER',
            cache = cache,
            nonpadding=transformer.features_to_nonpadding(features, "targets"))
      logits = dp(self.pointer_top, body_outputs, features)[0]
      #logits = tf.Print(logits,[i,length[0],logits[0,:,:,:,0],logits[0,:,:,:,47481],logits[0,:,:,:,47482]])
      mask = tf.reshape(mask,common_layers.shape_list(logits))
      #logits = tf.Print(logits,[i,mask[0,:,:,:,0],mask[0,:,:,:,1],mask[0,:,:,:,47481]])
      logits += tf.cast(mask,dtype=logits.dtype)*common_attention.large_compatible_negative(logits.dtype)
      logits = tf.clip_by_value(logits,common_attention.large_compatible_negative(logits.dtype),1e8)
      #logits = tf.Print(logits,[i,logits[0,:,:,:,0],logits[0,:,:,:,47481]])
      ret = tf.squeeze(logits, axis=[1, 2, 3])
      return ret, cache

    sos_id = self.get_decode_start_id() or 0
    eos_id = self.get_decode_end_id() or beam_search.EOS_ID
    temperature = features.get("sampling_temp",
                               getattr(hparams, "sampling_temp", 0.0))
    top_k = features.get("sampling_keep_top_k",
                         getattr(hparams, "sampling_keep_top_k", -1))

    ret = transformer.fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        init_cache_fn=self._init_cache_fn,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length,
        sos_id=sos_id,
        eos_id=eos_id,
        cache=cache)
    nonpadding=tf.cast(nonpadding,dtype=ret["outputs"].dtype)
    ret["outputs"]=ret["outputs"]*nonpadding+(1-nonpadding)
    return ret

  def pointer_top(self, body_output, features):
    output_shape = common_layers.shape_list(body_output) #only take 3d
    body_output = tf.reshape(body_output,output_shape[:2]+[output_shape[-1]])
    labels = features['targets']
    #labels = tf.Print(labels,[labels[0]],message='before_add: ',summarize=256)
    labels = add_eos_to_target_ids(labels)
    #labels = tf.Print(labels,[labels[0]],message='added: ',summarize=256)
    l_shape = common_layers.shape_list(labels)
    B = output_shape[0]//l_shape[0]
    labels = beam_search._expand_to_beam_size(labels,B)
    labels = tf.reshape(labels,[output_shape[0],l_shape[1]])
    output_shape[1] = l_shape[1]
    max_val = tf.reduce_max(labels)+1 # bigger than all tgt ids batchwise
    padding = tf.cast(tf.equal(labels,0),dtype=labels.dtype)
    labels += padding*tf.ones_like(labels)*max_val
    labels, ind = tf.nn.top_k(-labels, output_shape[1])
    labels = -labels#*(1-padding) # labels are sorted target ids with 0s turned max_val+1
    di = tf.reshape(self.decoder_input,[-1,output_shape[-1]])
    idv=tf.reshape(tf.tile(tf.reshape(tf.range(output_shape[0]),[-1,1]),[1,output_shape[1]])*output_shape[1]+ind,[-1])
    #idv=tf.reshape(tf.tile(tf.reshape(tf.range(output_shape[0]),[-1,1]),[1,output_shape[1]-1])*output_shape[1]+ind[:,1:],[-1])
    decoder_input = tf.gather(di,idv)
    #decoder_input = tf.Print(decoder_input,[idv[:60]],summarize=256,message='idv: ')
    decoder_input = tf.reshape(decoder_input,[output_shape[0],output_shape[1],output_shape[-1]])
    feature_name = 'targets'
    modality = self._problem_hparams.modality[feature_name]
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor

    name = self._hparams.name.get(
        feature_name,
        modalities.get_name(modality))(self._hparams, vocab_size)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      idx = tf.ones([output_shape[0],1],dtype=tf.int32)
      eos = select_symbol_top(idx,self._hparams,vocab_size)
      eos = tf.reshape(eos,[output_shape[0],1,output_shape[-1]])
    #vocab = eos ; input, input = bos ; encoder_output
    #decoder_input = tf.Print(decoder_input,[tf.shape(decoder_input)],message='shape1: ')
    vocab = tf.concat([eos,decoder_input[:,1:]],1)
    #vocab = tf.Print(vocab,[tf.shape(vocab)],message='shape1: ')
    #vocab = decoder_input
    vocab *= (1.0 - tf.tile(tf.cast(tf.expand_dims(padding,-1),dtype=vocab.dtype),[1,1,output_shape[-1]])) # padding embedding is all zeroed
    logits, mask = aligner_dot(body_output,vocab)
    #logits = tf.Print(logits,[logits[0],mask[0]],summarize=256)
    # logits = [B, T, T]
    # we now rearrange this sparse tensor into vocab_size
    T = common_layers.shape_list(logits)[1]
    ind = tf.reshape(tf.tile(tf.reshape(tf.range(output_shape[0]),[-1,1,1]),[1,T,output_shape[1]]),[-1])
    ind1 = tf.reshape(tf.tile(tf.reshape(tf.range(T),[1,-1,1]),[output_shape[0],1,output_shape[1]]),[-1])
    labels = tf.reshape(tf.tile(tf.expand_dims(labels,1),[1,T,1]),[-1])
    ind = tf.stack([ind,ind1,labels],-1)
    ind1 = tf.concat([[tf.zeros_like(ind[0])],ind[:-1]],0)
    dup_pad = tf.cast(tf.equal(tf.reduce_sum(ind-ind1,1),0),dtype=labels.dtype)
    labels = labels * (1-dup_pad) + dup_pad * max_val
    #labels = tf.Print(labels,[labels[:60]],message='dup_lbs: ',summarize=256)
    gather_pad = tf.cast(tf.equal(labels,max_val),dtype=labels.dtype)
    to_gather_ids = (1-gather_pad)*tf.cast(tf.range(output_shape[0]*T*output_shape[1]),dtype=labels.dtype)-gather_pad
    gather_sum = output_shape[0]*output_shape[1]*T-tf.reduce_sum(gather_pad)
    to_gather_ids = tf.nn.top_k(to_gather_ids,gather_sum)[0]
    to_gather_ids = tf.reshape(tf.reverse_sequence(tf.expand_dims(to_gather_ids,0),[gather_sum],1,0),[-1])
    ind = tf.gather(ind, to_gather_ids)
    labels = tf.gather(labels, to_gather_ids)
    #logits = tf.Print(logits,[labels[:64]],summarize=256,message='final_labels: ')
    logits = tf.gather(tf.reshape(logits,[-1]), to_gather_ids)
    ldtype = logits.dtype
    logits = tf.sparse.SparseTensor(tf.cast(ind,dtype=tf.int64),logits,[output_shape[0],T,vocab_size])
    # use one_hot to create window for valid items and clip mask the rest 
    logits = tf.sparse.to_dense(logits)
    logits = tf.reshape(logits,[output_shape[0],T,1,1,vocab_size])
    logits += tf.cast(tf.equal(logits,0),dtype=ldtype)*common_attention.large_compatible_negative(ldtype)
    logits = tf.clip_by_value(logits,common_attention.large_compatible_negative(ldtype),1e8)
    return logits

@registry.register_model
class nalm(transformer.Transformer):
  def __init__(self, *args, **kwargs):
    super(nalm, self).__init__(*args, **kwargs)
    self._prepare_decoder_fn = nalm_prepare_decoder
    self._hparams.has_input = self.has_input
    self.bi = False
    self.mnalm = False

  def bottom(self, features):
    if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
            self._hparams.mode != "attack"):
      features['targets']=features['ground_truth']
    #remove eos if any
    features['targets'] *= tf.cast(tf.greater(features['targets'],1),
                             dtype=features['targets'].dtype) 
    transformed_features = super(nalm, self).bottom(features)
    targets = transformed_features['targets']
    target_shape = common_layers.shape_list(targets)
    bos = get_bos(targets[:,0:1])
    bos = tf.reshape(bos, [target_shape[0],1,1,target_shape[-1]])
    transformed_features['targets'] = tf.concat([bos,targets],1)
    return transformed_features

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             mode='TRAIN',
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    input_shape = common_layers.shape_list(decoder_input)
    #save decoder_input
    self.decoder_input = decoder_input
    padding = common_attention.attention_bias_to_padding(decoder_self_attention_bias)
    if hparams.get('mask_window',None) is not None and self.mnalm:
      single_width = tf.cond(tf.less(input_shape[1],hparams.mask_window),
                          lambda: input_shape[1],
                          lambda: hparams.mask_window)
      dsad = common_attention.attention_bias_local(input_shape[1],
                                                   single_width,
                                                   single_width)
      dsad = tf.reshape(dsad,[1,1,input_shape[1],input_shape[1]])
      decoder_self_attention_bias = tf.tile(decoder_self_attention_bias,[1,1,input_shape[1],1])
      decoder_self_attention_bias += tf.tile(dsad,[input_shape[0],1,1,1])
      decoder_self_attention_bias = tf.clip_by_value(decoder_self_attention_bias, common_attention.large_compatible_negative(dsad.dtype),1)
    decoder_output = transformer.transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,**kwargs) 
    decoder_output*=tf.tile(tf.reshape(1.0-padding,input_shape[:2]+[1,1]),[1,1,1,input_shape[-1]])
    return decoder_output

  def infer(self, features,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    with tf.variable_scope(self.name):
      print([x for x in features])
      if self.has_input:
        features["target_space_id"] = features.get("target_space_id",tf.constant(0))
      else:
        features["targets"]=features["inputs"]
        features["ground_truth"]=features["inputs"]
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16) #ground_truth, input 

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        body_out = self.body(transformed_features)
      outputs, losses = self._normalize_body_output(body_out)

      #a = a_logits, b = a_mask
      logits, mask = self.pointer_top(outputs, features)
      A_mat2=None
      if self.bi:
        logits,l2=logits
        masks,m2=masks
      l_shape = common_layers.shape_list(logits)
      logits = tf.reshape(logits,[l_shape[0],l_shape[1],l_shape[-1]])
      m_shape=common_layers.shape_list(mask)
      mask = tf.reshape(mask,m_shape[:2]+[m_shape[-1]])
      A_matrix = tf.nn.softmax(logits)*tf.tile(tf.expand_dims(
                 mask[:,0],-1),[1,1,l_shape[1]])
      if self.bi:
        l2 = tf.reshape(l2,[l_shape[0],l_shape[1],l_shape[-1]])
        m2 = tf.reshape(m2,m_shape[:2]+[m_shape[-1]])
        A_mat2 = tf.nn.softmax(l2)*tf.tile(tf.expand_dims(
                 m2[:,0],-1),[1,1,l_shape[1]])

      inputs = tf.reshape(features["targets"],
                 common_layers.shape_list(features["targets"])[:2])
      nonpadding = tf.cast(tf.greater(inputs,1),dtype=inputs.dtype)
      nonpadding = tf.concat([nonpadding[:,0:1],nonpadding],-1)
      ret = combine_align(inputs,A_matrix,A_mat2,nonpadding,beam_size)
      ret = tf.squeeze(ret)
      return ret

  def pointer_top(self, body_output, features):
    output_shape = common_layers.shape_list(body_output) #only take 3d
    body_output = tf.reshape(body_output,output_shape[:2]+[output_shape[-1]])
    feature_name = 'targets'
    modality = self._problem_hparams.modality[feature_name]
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    name = self._hparams.name.get(
        feature_name,
        modalities.get_name(modality))(self._hparams, vocab_size)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      idx = tf.ones([output_shape[0],1],dtype=tf.int32)
      eos = select_symbol_top(idx,self._hparams,vocab_size)
      eos = tf.reshape(eos,[output_shape[0],1,output_shape[-1]])
    #vocab = eos ; input, input = bos ; encoder_output 
    vocab = tf.concat([eos,self.decoder_input[:,1:]],1)
    gt = features['targets'] 
    if features.get('ground_truth') is not None:
      gt = features['ground_truth']
    nonpadding = tf.cast(tf.greater(gt,1),dtype=vocab.dtype)
    nonpadding = tf.concat([tf.ones_like(nonpadding[:,0:1]), nonpadding],1)
    if self.bi:
      id1=tf.range(output_shape[-1]//2)*2
      id2=tf.range(output_shape[-1]//2)*2+1
      v1=tf.gather(vocab,id1,axis=-1)
      v2=tf.gather(vocab,id2,axis=-1)
      b1=tf.gather(body_output,id1,axis=-1)
      b2=tf.gather(body_output,id2,axis=-1)
      l1, m1 = aligner_dot(b1,v1,nonpadding)
      l2, m2 = aligner_dot(b2,v2,nonpadding)
      return (l1,l2),(m1,m2)
    logits, mask = aligner_dot(body_output,vocab,nonpadding)
    return logits, mask

  def loss(self, logits, features):
    inputs = features["targets"]
    logit_shape = common_layers.shape_list(logits)
    nonpadding = tf.cast(tf.greater(tf.squeeze(inputs),1),
                         dtype=inputs.dtype)
    nonpadding = tf.concat([tf.ones_like(nonpadding[:,0:1]),
                            nonpadding],1)
    a_label = tf.tile(tf.expand_dims(tf.cast(tf.range(
                logit_shape[1])+1,dtype=inputs.dtype),0),[logit_shape[0],1])
    a_label *= tf.cast(tf.concat([nonpadding[:,1:],
                tf.zeros_like(nonpadding[:,0:1])],1),dtype=a_label.dtype)
    a_label = tf.one_hot(a_label,logit_shape[1])
    a_label *= tf.cast(tf.tile(tf.expand_dims(nonpadding,-1),
               [1,1,logit_shape[1]]),dtype=a_label.dtype)
    return tf.nn.softmax_cross_entropy_with_logits_v2(
             logits=logits,labels=a_label) 

  def top(self, outputs, features):
    return self.pointer_top(outputs, features)[0]

@registry.register_model
class nalm_pos(nalm):
  def __init__(self, *args, **kwargs):
    super(nalm_pos, self).__init__(*args, **kwargs)
    self._prepare_decoder_fn = nalm_pos_prepare_decoder

  def bottom(self, features):
    #remove eos if any
    features['targets'] *= tf.cast(tf.greater(features['targets'],1),
                             dtype=features['targets'].dtype) 
    transformed_features = super(nalm, self).bottom(features)
    targets = transformed_features['targets']
    target_shape = common_layers.shape_list(targets)
    bos = get_bos(targets[:,0:1])
    bos = tf.reshape(bos, [target_shape[0],1,1,target_shape[-1]])
    transformed_features['targets'] = tf.concat([bos,targets],1)
    return transformed_features

  def loss(self, logits, features):
    logits_bwd = None
    if self.bi:
      assert len(logits)==2
      logits,logits_bwd = logits
    logit_shape = common_layers.shape_list(features['targets'])
    logits = tf.reshape(logits,[logit_shape[0],logit_shape[1]+1,-1])
    targets = tf.reshape(features['targets'],logit_shape[:2])
    ground_truth = tf.reshape(features['ground_truth'],logit_shape[:2])
    tgt = tf.tile(tf.expand_dims(targets,-1),[1,1,logit_shape[1]])
    gt = tf.tile(tf.expand_dims(ground_truth,1),[1,logit_shape[1],1])
    align_matrix = tf.cast(tf.equal(tgt,gt),dtype=targets.dtype)
    amat=-batch_upscale(align_matrix)
    amat += tf.cast(tf.equal(amat,0),dtype=amat.dtype)*logit_shape[1]*10
    ids = tf.map_fn(lambda x: dummy_wrap_scipy_hungarian(x), amat)
    nonpadding=tf.cast(tf.greater(ground_truth,1),dtype=ground_truth.dtype)
    eos = tf.expand_dims(tf.reduce_sum(nonpadding,-1),1)
    bos = tf.tile(tf.constant([[0]]),[logit_shape[0],1])
    tgt=tf.concat([bos,ids+1],-1)
    gt=tf.concat([eos,ids*nonpadding+(ids+1)*(1-nonpadding)],-1)
    tgt = tf.tile(tf.expand_dims(tgt,-1),[1,1,tf.shape(tgt)[1]])
    gt = tf.tile(tf.expand_dims(gt,1),[1,tf.shape(gt)[1],1])
    nonpadding = tf.concat([tf.ones_like(nonpadding[:,0:1]),nonpadding],1)
    weights = tf.cast(tf.reduce_sum(nonpadding),dtype=logits.dtype)
    a_label = tf.cast(tf.equal(tgt,gt),dtype=targets.dtype) #shifted labels
    a_label *= tf.cast(tf.tile(tf.expand_dims(nonpadding,-1),[1,1,logit_shape[1]+1]),dtype=a_label.dtype)
    xent= tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=a_label)
    if logits_bwd is not None:
      a_label_bwd = tf.cast(tf.equal(gt,tgt),dtype=targets.dtype)
      a_label_bwd *= tf.cast(tf.tile(tf.expand_dims(nonpadding,-1),[1,1,logit_shape[1]+1]),dtype=a_label_bwd.dtype)
      xent_bwd = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_bwd,labels=a_label_bwd)
      return (tf.reduce_sum(xent)+tf.reduce_sum(xent_bwd))/weights
    return tf.reduce_sum(xent)/weights

@registry.register_model
class reordernatr(transformer.Transformer):
  def  __init__(self, *args, **kwargs):
    super(reordernatr, self).__init__(*args, **kwargs)
    self._prepare_decoder_fn = nalm_pos_prepare_decoder
    self._hparams.has_input = self.has_input

  def loss(self, logits, features):
    features['targets']=features['ground_truth']
    return super(reordernatr, self).loss(logits,features)

  def pointer_top(self, body_output, features):
    output_shape = common_layers.shape_list(body_output) #only take 3d
    body_output = tf.reshape(body_output,output_shape[:2]+[output_shape[-1]])
    feature_name = 'targets'
    modality = self._problem_hparams.modality[feature_name]
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    name = self._hparams.name.get(
        feature_name,
        modalities.get_name(modality))(self._hparams, vocab_size)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      log_info("Transforming body output with %s.top", name)
      target_shape = common_layers.shape_list(features['targets'])
      idx = tf.reshape(features['targets'],[-1,1])
      vocab = select_symbol_top(idx,self._hparams,vocab_size)
      vocab = tf.reshape(vocab,target_shape[:2]+[-1])
      gt = features['targets'] 
      if features.get('ground_truth') is not None:
        gt = features['ground_truth']
      nonpadding = tf.cast(tf.greater(gt,1),dtype=vocab.dtype)
      logits, mask = aligner_dot(body_output,vocab,nonpadding)
    return logits, mask

  def infer(self, features,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    self.beam_size=beam_size
    log_info(' '.join([l for l in features]))
    with tf.variable_scope(self.name):
      if self.has_input:
        features["target_space_id"] = features.get("target_space_id",tf.constant(0))
      else:
        features["targets"]=features["inputs"]
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16) #ground_truth, input 

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        outputs = self.body(transformed_features)

      logits,mask = self.pointer_top(outputs, features)
      A_matrix = tf.nn.softmax(logits)*tf.tile(tf.expand_dims(tf.squeeze(mask)[:,0],-1),[1,1,common_layers.shape_list(logits)[1]])
      inputs = tf.squeeze(features["targets"])
      nonpadding = tf.cast(tf.greater(inputs,0),dtype=inputs.dtype)
      ret = perm_align(inputs,A_matrix,nonpadding)
      ret = tf.squeeze(ret)
      return ret

@registry.register_model
class snatr(reordernatr):
  def __init__(self, *args, **kwargs):
    super(snatr, self).__init__(*args, **kwargs)
    self._decoder_function = regularized_decoder

  def decode(self, 
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    """Decode Transformer outputs, see transformer_decode."""
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      self.layer_output_cache = {}
      kwargs.update({'layer_output_cache': self.layer_output_cache})
    return transformer.transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
        **kwargs)

  def body(self, features):
    log_info(' '.join([l for l in features]))
    x=features['target_space_id']
    ret = super(snatr, self).body(features)
    if self._hparams.mode != tf.estimator.ModeKeys.TRAIN:
      return ret
    if isinstance(ret,tuple) or isinstance(ret,list):
      ret[0] = {"targets": ret[0]}
      ret[0].update(self.layer_output_cache)
    else:
      ret = {"targets": ret}
      ret.update(self.layer_output_cache)
    return ret

  def top(self, body_output, features):
    if isinstance(body_output, dict):
      logits = {}
      for k, v in six.iteritems(body_output):
        logits[k] = self._top_single(v, 'targets', features)
      return logits
    else:
      return self._top_single(body_output, "targets", features)

  def loss(self, logits, features):
    log_info(' '.join([l for l in features]))
    # we do not train with pointer loss
    ground_truth = features["ground_truth"]
    ground_truth = tf.expand_dims(tf.expand_dims(ground_truth,-1),-1)
    if isinstance(logits, dict):
      losses = {}
      for k,v in six.iteritems(logits):
        #v = tf.Print(v,[tf.shape(v)],summarize=5, message=k)
        losses[k] = self._loss_single(v,
                                      "targets",
                                      ground_truth,
                                      weights=features.get("targets_mask"))
        n, d = losses[k]
        if common_layers.should_generate_summaries():
          tf.summary.scalar(k + "_loss", n / d)
          tf.summary.scalar(k + "_loss_num", n)
          tf.summary.scalar(k + "_loss_den", d)
          if getattr(self.hparams, "visualize_logits_histogram", False):
            hist = tf.summary.histogram
            hist(k + "_predict", tf.argmax(tf.squeeze(v), axis=-1))
            hist(k + "_targets", features[k])

      return tf.add_n([n / d for n, d in losses.values()])
    else:
      return self._loss_single(
          logits,
          "targets",
          ground_truth,
          weights=features.get("targets_mask"))

@registry.register_model
class alignartr(transformer.Transformer):
  def __init__(self, *args, **kwargs):
    super(alignartr, self).__init__(*args, **kwargs)
    self._prepare_decoder_fn = nalm_pos_prepare_decoder
    self._hparams.has_input = self.has_input

  def permutation_top(self, outputs, features):
    targets = features['targets']
    nonpadding = tf.squeeze(tf.cast(tf.greater(targets,1),dtype=outputs.dtype))
    return permutation_weights(outputs, nonpadding, self._hparams)

  def top(self, outputs, features):
    #instead of regular top, we do the permutation thing
    logits = self.permutation_top(outputs,features)[0]
    return logits

  def loss(self, logits, features):
    #we minimize KL divergence between permutation matrix and ground truth
    log_info(' '.join([l for l in features]))
    assert 'ground_truth' in features
    targets = features['targets']
    ground_truth = features['ground_truth']
    logits_shape = common_layers.shape_list(logits)
    nonpadding = tf.cast(tf.greater(tf.squeeze(targets),1),dtype=logits.dtype)
    ground_truth_matrix = compute_alignment_matrix(targets, ground_truth, nonpadding)
    loss = tf.reduce_sum(klx(ground_truth_matrix,logits)*nonpadding)
    return loss

  def infer(self, features,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    self.beam_size=beam_size
    with tf.variable_scope(self.name):
      if self.has_input:
        features["target_space_id"] = features.get("target_space_id",tf.constant(0))
      else:
        features["targets"]=features["inputs"]
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16) #ground_truth, input 

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        outputs = self.body(transformed_features)

      logits,mask = self.permutation_top(outputs, features)
      #logits = tf.Print(logits,[logits[-1],mask[-1]],message='S: ',summarize=10000)
      A_matrix = logits*tf.tile(tf.expand_dims(tf.squeeze(mask)[:,0],-1),[1,1,common_layers.shape_list(logits)[1]])
      inputs = tf.squeeze(features["targets"])
      nonpadding = tf.cast(tf.greater(inputs,1),dtype=inputs.dtype)
      ret = perm_align(inputs,A_matrix,nonpadding)
      ret = tf.squeeze(ret)
      return ret

@registry.register_model
class distortionr(transformer.Transformer):
  def __init__(self, *args, **kwargs):
    super(distortionr, self).__init__(*args, **kwargs)
    self._hparams.has_input = self.has_input
    self._prepare_decoder_fn = nalm_pos_prepare_decoder

  def distortion_predictor(self, body_output, bias, hparams):
    body_output=common_layers.flatten4d3d(body_output)
    body_output_shape = common_layers.shape_list(body_output)
    k = 100 #max_relative_position
    with tf.variable_scope('distortion_layer'):
      y = common_attention.multihead_attention(
          body_output,
          None,
          bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size,
          hparams.num_heads,
          hparams.attention_dropout,
          attention_type="dot_product_relative",
          max_relative_position=k,
          max_length=hparams.get("max_length"),
          activation_dtype=hparams.get("activation_dtype", "float32"),
          weight_dtype=hparams.get("weight_dtype", "float32"),
          area_key_mode=hparams.get("area_key_mode", "none"),
          area_value_mode=hparams.get("area_value_mode", "none"),
          training=(hparams.get("mode", tf.estimator.ModeKeys.TRAIN)
            == tf.estimator.ModeKeys.TRAIN))

    with tf.variable_scope('distortion_predictor'):
      var = tf.get_variable('repos',[k*2+1,hparams.hidden_size],
                      initializer=tf.random_normal_initializer(0.0, hparams.hidden_size**-0.5))
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
      logits = tf.matmul(body_output, var, transpose_b=True)
      return tf.reshape(logits, body_output_shape[:-1] + [1, k*2+1])

  def top(self, output, features):
    # here we reformat the logits to realign it for hungarian
    targets = features['targets']
    target_shape = common_layers.shape_list(targets)
    nonpadding = tf.reshape(tf.cast(tf.greater(targets,0),dtype=output.dtype),target_shape[:2])
    bias = common_attention.attention_bias_ignore_padding(1.0-nonpadding)
    logits = self.distortion_predictor(output, bias, self._hparams)
    logit_shape = common_layers.shape_list(logits)
    #logits = tf.Print(logits,[logit_shape],message='l_shape: ',summarize=10)
    zero_idx = logit_shape[-1]//2
    length = tf.reduce_sum(tf.cast(nonpadding,dtype=tf.int32),1)
    max_length = logit_shape[1]#tf.reduce_max(length)
    base = tf.tile(tf.expand_dims(tf.range(logit_shape[-1]),0),[logit_shape[0]*logit_shape[1],1])
    pidx = tf.reshape(tf.tile(tf.expand_dims(tf.range(logit_shape[1]),0),[logit_shape[0],1]),[-1,1])
    p2 = tf.tile(tf.reshape(tf.tile(tf.expand_dims(zero_idx+length,1),[1,logit_shape[1]]),[-1,1])-pidx,[1,logit_shape[-1]])
    p1 = tf.tile(zero_idx-pidx,[1,logit_shape[-1]])
    m = tf.cast(tf.logical_and(tf.greater_equal(base,p1),tf.less(base,p2)),dtype=logits.dtype)
    m,idx = tf.nn.top_k(m, max_length)
    m = tf.reshape(m, [logit_shape[0],logit_shape[1],max_length])
    idx += tf.reshape(tf.cast(tf.tile(tf.expand_dims(tf.range(logit_shape[0]),1),[1,logit_shape[1]*max_length])*logit_shape[-1],dtype=idx.dtype),tf.shape(idx))
    #idx = tf.Print(idx,[idx],summarize=25600, message='idx')
    logits = tf.gather_nd(tf.reshape(logits,[-1]),tf.reshape(idx,[-1,1]))
    logits = tf.reshape(logits,[logit_shape[0],logit_shape[1],max_length])
    #logits = tf.Print(logits,[tf.shape(logits)],message='l1_shape: ',summarize=10)
    return logits * m + (1-m)* common_attention.large_compatible_negative(logits.dtype)

  def loss(self, logits, features):
    #TODO, adapt below for use
    assert 'ground_truth' in features
    targets = features['targets']
    ground_truth = features.get('ground_truth',None)
    logits_shape = common_layers.shape_list(logits)
    #targets=tf.Print(targets,[logits_shape,tf.shape(targets),tf.shape(ground_truth)],summarize=20)
    nonpadding = tf.cast(tf.greater(tf.squeeze(targets),1),dtype=logits.dtype)
    labels = tf.expand_dims(tf.linalg.diag(nonpadding),1)
    if ground_truth is not None:
      labels = compute_alignment_matrix(targets, ground_truth, nonpadding)
    #labels = tf.Print(labels,[tf.reduce_any(tf.is_nan(labels))],message='labels: ')
    confidence = 1.0 - self._hparams.label_smoothing
    low_confidence = (1.0 - confidence) / common_layers.to_float(logits_shape[-1] - 1)
    normalizing = -(
        confidence * tf.log(confidence) + common_layers.to_float(logits_shape[-1] - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))
    #logits = tf.Print(logits,[tf.shape(logits),tf.shape(labels),tf.shape(normalizing)],summarize=150)
    xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels) - normalizing
    #xent=tf.Print(xent,[tf.reduce_any(tf.is_nan(xent*nonpadding))],message='xent: ')
    return tf.reduce_sum(xent*nonpadding),tf.reduce_sum(nonpadding)

  def infer(self, features,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    self.beam_size=beam_size
    with tf.variable_scope(self.name):
      if self.has_input:
        features["target_space_id"] = features.get("target_space_id",tf.constant(0))
      else:
        features["targets"]=features["inputs"]
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16) #ground_truth, input 

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        outputs = self.body(transformed_features)

      logits = self.top(outputs, features)
      inputs = tf.squeeze(features["targets"])
      nonpadding = tf.cast(tf.greater(inputs,1),dtype=inputs.dtype)
      A_matrix = tf.nn.softmax(logits)*tf.cast(tf.tile(tf.expand_dims(nonpadding,-1),[1,1,common_layers.shape_list(logits)[-1]]),dtype=logits.dtype)
      ret = perm_align(inputs,A_matrix,nonpadding)
      ret = tf.squeeze(ret)
      return ret

@registry.register_hparams
def mnalm_multistep8(): #with mask
  hparams = transformer.transformer_base_multistep8()
  hparams.add_hparam("mask_window",1)
  return hparams


