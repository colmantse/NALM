#execute this script to train and validate model performance on dataset, comment out the train and validation section and uncomment the evaluate section to test final model performance. Our codes depend on the tensor2tensor library. Please visit https://github.com/tensorflow/tensor2tensor for more detail. 

problem=ptb2016_shuffle #see registered problem at usr/t2t_usr/ptb2016.py for more detail
model=nalm #see registered model at usr/t2t_usr/nalm.py for more detail
hparams=transformer_base_multistep8 #adjusted based on the number of available gpus, current setting is for dual-gpu machine, see tensor2tensor/models/transformer.py for more detail
beam_size=1 #for non-autoregressive model, please set this as 1, for autoregressive model, adjust according to need.

train_dir=./t2t_train/reorder/$problem/$model'_'$hparams
trans_dir_base=$model'_'$hparams/$problem/$beam_size
translation_dir=test/$trans_dir_base

source_dir=./tmp/t2t_datagen/ptb2016/ptb2016.valid.txt #ground truth validation data
source_ref=./tmp/t2t_datagen/ptb2016/ptb2016.valid.fullyshuffled.txt #validation input
test_dir=./tmp/t2t_datagen/ptb2016/ptb2016.test.txt #ground truth test data
test_ref=./tmp/t2t_datagen/ptb2016/ptb2016.test.fullyshuffled.txt #test input

steps=100000
min_steps=9999
best_score=93644 #modify this based on validation results before running the evaluation.

##Training and validation
t2t-trainer --t2t_usr_dir=./usr/t2t_usr --data_dir=./t2t_data --problem=$problem --model=$model --hparams_set=$hparams --output_dir=$train_dir --train_steps=$steps --worker_gpu=2 --worker_gpu_memory_fraction=0.9 --save_checkpoints_secs=3600 --tmp_dir=./tmp/t2t_datagen --schedule='train' --generate_data

t2t-translate-all --t2t_usr_dir=./usr/t2t_usr --model=$model --hparams_set=$hparams --tmp_dir=./tmpt2t_datagen --source=$source_ref --translations_dir=$translation_dir --model_dir=$train_dir --problem=$problem --data_dir=./t2t_data --min_steps=$min_steps --beam_size=$beam_size

t2t-bleu --translations_dir=$translation_dir --reference=$source_dir --event_dir=event/$trans_dir_base

##Evaluate
#t2t-decoder --t2t_usr_dir=./usr/t2t_usr --output_dir=$train_dir --data_dir=./t2t_data --problem=$problem --model=$model --hparams_set=$hparams --checkpoint_path=$train_dir/model.ckpt-$best_score --decode_to_file=$translation_dir/best --eval_use_test_set=True --keep_timestamp --decode_hparams="batch_size=32, beam_size=$beam_size" --decode_from_file=$test_ref

#python compute_meteor.py $translation_dir/best $test_dir
#t2t-bleu --translation=$translation_dir/best --reference=$test_dir
