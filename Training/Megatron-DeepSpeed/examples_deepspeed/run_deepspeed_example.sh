#!/bin/bash
set -ex

BASE_PATH=/home/mm12318/HPML/Proj/Megatron-DeepSpeed;
DATA_PATH=/home/mm12318/HPML/Proj;
DS_CONFIG=ds_config.json;

TP=2;
PP=1;
NLAYERS=32;
HIDDEN=2560;

GLOBAL_BATCH=16;
MICRO_BATCH=8;

ZERO_STAGE=3;

OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH};
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR;

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "autotuning": {
    "enabled": true,
    "arg_mappings": {
      "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
      "gradient_accumulation_steps ": "--gradient_accumulation_steps"
    }
  },

  "wall_clock_breakdown" : true,
  "quantize_training": {
      "enabled": true,
      "quantize_verbose": true,
      "quantizer_kernel": true,
      "quantize-algo": {
        "q_type": "symmetric"
      },
      "quantize_bits": {
        "start_bits": 16,
        "target_bits": 8
      },
      "quantize_schedule": {
        "quantize_period": 400,
        "schedule_offset": 0
      },
      "quantize_groups": 8
    }
}
EOT

export NCCL_DEBUG=warn ;

ds_args="";
ds_args=" --deepspeed ${ds_args}";
ds_args=" --no-pipeline-parallel ${ds_args}" ;
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}";
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}";
ds_args=" --deepspeed-activation-checkpointing ${ds_args}";


deepspeed pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --loss-scale 12 \
    --max-position-embeddings 2048 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 1000 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path /home/mm12318/HPML/Proj/output_data_text_document \
    --vocab-file $BASE_PATH/vocab.json \
    --merge-file $BASE_PATH/merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    --checkpoint-activations \
    --tensorboard-dir $OUTPUT_DIR \
    $ds_args \
    --exit-interval 5000 | tee ${OUTPUT_DIR}/output.log

