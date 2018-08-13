# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/bin/bash
# The following scripts assumes there are at least 6 GPUs on the machine.
# Otherwise you need to change the device ID and run them on different machines.

# ==============================================================================
#                              Baseline
# ==============================================================================

# CIFAR-10 baseline
python code/cifar_train_baseline.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/0.2 \
  --train_log_dir=cifar_models/cifar10/resnet/0.2/baseline/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 > stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
 --data_dir=data/cifar10/val/ \
 --checkpoint_dir=cifar_models/cifar10/resnet/0.2/baseline/train \
 --eval_dir=cifar_models/cifar10/resnet/0.2/baseline/eval_val \
 --studentnet=resnet101 --device_id=1 > stdout1.txt 2>stderr1.txt &

python code/cifar_train_baseline.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/0.4 \
  --train_log_dir=cifar_models/cifar10/resnet/0.4/baseline/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 > stdout2.txt 2>stderr2.txt  &

python code/cifar_eval.py --dataset_name=cifar10 \
 --data_dir=data/cifar10/val/ \
 --checkpoint_dir=cifar_models/cifar10/resnet/0.4/baseline/train \
 --eval_dir=cifar_models/cifar10/resnet/0.4/baseline/eval_val \
 --studentnet=resnet101 --device_id=3 > stdout3.txt 2>stderr3.txt &

python code/cifar_train_baseline.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/0.8 \
  --train_log_dir=cifar_models/cifar10/resnet/0.8/baseline/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 > stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
 --data_dir=data/cifar10/val/ \
 --checkpoint_dir=cifar_models/cifar10/resnet/0.8/baseline/train \
 --eval_dir=cifar_models/cifar10/resnet/0.8/baseline/eval_val \
 --studentnet=resnet101 --device_id=5 > stdout5.txt 2>stderr5.txt &

# CIFAR-100 baseline
python code/cifar_train_baseline.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/0.2 \
  --train_log_dir=cifar_models/cifar100/resnet/0.2/baseline/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
 --data_dir=data/cifar100/val/ \
 --checkpoint_dir=cifar_models/cifar100/resnet/0.2/baseline/train \
 --eval_dir=cifar_models/cifar100/resnet/0.2/baseline/eval_val \
 --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_baseline.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/0.4 \
  --train_log_dir=cifar_models/cifar100/resnet/0.4/baseline/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
 --data_dir=data/cifar100/val/ \
 --checkpoint_dir=cifar_models/cifar100/resnet/0.4/baseline/train \
 --eval_dir=cifar_models/cifar100/resnet/0.4/baseline/eval_val \
 --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_baseline.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/0.8 \
  --train_log_dir=cifar_models/cifar100/resnet/0.8/baseline/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
 --data_dir=data/cifar100/val/ \
 --checkpoint_dir=cifar_models/cifar100/resnet/0.8/baseline/train \
 --eval_dir=cifar_models/cifar100/resnet/0.8/baseline/eval_val \
 --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &


# ==============================================================================
#                          Self-paced MentorNet
# ==============================================================================

# CIFAR-10
python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/self_paced/self_paced_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar10/0.2 \
  --train_log_dir=cifar_models/cifar10/resnet/0.2/selfpaced/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.2/selfpaced/train \
  --eval_dir=cifar_models/cifar10/resnet/0.2/selfpaced/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/self_paced/self_paced_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar10/0.4 \
  --train_log_dir=cifar_models/cifar10/resnet/0.4/selfpaced/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.4/selfpaced/train \
  --eval_dir=cifar_models/cifar10/resnet/0.4/selfpaced/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/self_paced/self_paced_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar10/0.8 \
  --train_log_dir=cifar_models/cifar10/resnet/0.8/selfpaced/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.8/selfpaced/train \
  --eval_dir=cifar_models/cifar10/resnet/0.8/selfpaced/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &


# CIFAR-100
python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/self_paced/self_paced_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar100/0.2 \
  --train_log_dir=cifar_models/cifar100/resnet/0.2/selfpaced/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.2/selfpaced/train \
  --eval_dir=cifar_models/cifar100/resnet/0.2/selfpaced/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/self_paced/self_paced_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar100/0.4 \
  --train_log_dir=cifar_models/cifar100/resnet/0.4/selfpaced/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.4/selfpaced/train \
  --eval_dir=cifar_models/cifar100/resnet/0.4/selfpaced/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/self_paced/self_paced_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar100/0.8 \
  --train_log_dir=cifar_models/cifar100/resnet/0.8/selfpaced/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.8/selfpaced/train \
  --eval_dir=cifar_models/cifar100/resnet/0.8/selfpaced/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &


# ==============================================================================
#                          Focal Loss MentorNet
# ==============================================================================

# CIFAR-10
python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/focal_loss/focal_loss_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar10/0.2 \
  --train_log_dir=cifar_models/cifar10/resnet/0.2/focalloss/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.2/focalloss/train \
  --eval_dir=cifar_models/cifar10/resnet/0.2/focalloss/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/focal_loss/focal_loss_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar10/0.4 \
  --train_log_dir=cifar_models/cifar10/resnet/0.4/focalloss/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.4/focalloss/train \
  --eval_dir=cifar_models/cifar10/resnet/0.4/focalloss/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/focal_loss/focal_loss_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar10/0.8 \
  --train_log_dir=cifar_models/cifar10/resnet/0.8/focalloss/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.8/focalloss/train \
  --eval_dir=cifar_models/cifar10/resnet/0.8/focalloss/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &


# CIFAR-100
python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/focal_loss/focal_loss_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar100/0.2 \
  --train_log_dir=cifar_models/cifar100/resnet/0.2/focalloss/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.2/focalloss/train \
  --eval_dir=cifar_models/cifar100/resnet/0.2/focalloss/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/focal_loss/focal_loss_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar100/0.4 \
  --train_log_dir=cifar_models/cifar100/resnet/0.4/focalloss/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.4/focalloss/train \
  --eval_dir=cifar_models/cifar100/resnet/0.4/focalloss/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/focal_loss/focal_loss_mentornet \
  --loss_p_precentile=0.7 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.0,100" \
  --data_dir=data/cifar100/0.8 \
  --train_log_dir=cifar_models/cifar100/resnet/0.8/focalloss/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.8/focalloss/train \
  --eval_dir=cifar_models/cifar100/resnet/0.8/focalloss/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &

# ==============================================================================
#                          MentorNet PD
# ==============================================================================
# CIFAR-10
python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/mentornet_pd1_g_1/mentornet_pd \
  --loss_p_precentile=0.75 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.5,17,0.05,83" \
  --data_dir=data/cifar10/0.2 \
  --train_log_dir=cifar_models/cifar10/resnet/0.2/mentornet_pd1_g_1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.2/mentornet_pd1_g_1/train \
  --eval_dir=cifar_models/cifar10/resnet/0.2/mentornet_pd1_g_1/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/mentornet_pd1_g_1/mentornet_pd \
  --loss_p_precentile=0.75 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.5,17,0.05,83" \
  --data_dir=data/cifar10/0.4 \
  --train_log_dir=cifar_models/cifar10/resnet/0.4/mentornet_pd1_g_1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.4/mentornet_pd1_g_1/train \
  --eval_dir=cifar_models/cifar10/resnet/0.4/mentornet_pd1_g_1/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/mentornet_pd1_g_1/mentornet_pd \
  --loss_p_precentile=0.5 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.5,17,0.05,83" \
  --data_dir=data/cifar10/0.8 \
  --train_log_dir=cifar_models/cifar10/resnet/0.8/mentornet_pd1_g_1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.8/mentornet_pd1_g_1/train \
  --eval_dir=cifar_models/cifar10/resnet/0.8/mentornet_pd1_g_1/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &

# CIFAR-100
python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/mentornet_pd1_g_1/mentornet_pd \
  --loss_p_precentile=0.75 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.5,17,0.05,83" \
  --data_dir=data/cifar100/0.2 \
  --train_log_dir=cifar_models/cifar100/resnet/0.2/mentornet_pd1_g_1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.2/mentornet_pd1_g_1/train \
  --eval_dir=cifar_models/cifar100/resnet/0.2/mentornet_pd1_g_1/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/mentornet_pd1_g_1/mentornet_pd \
  --loss_p_precentile=0.75 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.5,17,0.05,83" \
  --data_dir=data/cifar100/0.4 \
  --train_log_dir=cifar_models/cifar100/resnet/0.4/mentornet_pd1_g_1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.4/mentornet_pd1_g_1/train \
  --eval_dir=cifar_models/cifar100/resnet/0.4/mentornet_pd1_g_1/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/mentornet_pd1_g_1/mentornet_pd \
  --loss_p_precentile=0.5 \
  --nofixed_epoch_after_burn_in \
  --burn_in_epoch=0 \
  --example_dropout_rates="0.5,17,0.05,83" \
  --data_dir=data/cifar100/0.8 \
  --train_log_dir=cifar_models/cifar100/resnet/0.8/mentornet_pd1_g_1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.8/mentornet_pd1_g_1/train \
  --eval_dir=cifar_models/cifar100/resnet/0.8/mentornet_pd1_g_1/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &

# ==============================================================================
#                          MentorNet DD
# ==============================================================================
# CIFAR-10
python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/nl_0.2_percentile_90/mentornet_dd \
  --loss_p_precentile=0.8 \
  --fixed_epoch_after_burn_in \
  --burn_in_epoch=18 \
  --example_dropout_rates="0.5,17,0.05,78,0.9,5" \
  --data_dir=data/cifar10/0.2 \
  --train_log_dir=cifar_models/cifar10/resnet/0.2/mentornet_dd1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.2/mentornet_dd1/train \
  --eval_dir=cifar_models/cifar10/resnet/0.2/mentornet_dd1/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/nl_0.2_percentile_90/mentornet_dd \
  --loss_p_precentile=0.6 \
  --fixed_epoch_after_burn_in \
  --burn_in_epoch=18 \
  --example_dropout_rates="0.5,17,0.05,78,0.9,5" \
  --data_dir=data/cifar10/0.4 \
  --train_log_dir=cifar_models/cifar10/resnet/0.4/mentornet_dd1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.4/mentornet_dd1/train \
  --eval_dir=cifar_models/cifar10/resnet/0.4/mentornet_dd1/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar10 \
  --trained_mentornet_dir=mentornet_models/models/nl_0.2_percentile_90/mentornet_dd \
  --loss_p_precentile=0.5 \
  --fixed_epoch_after_burn_in \
  --burn_in_epoch=18 \
  --example_dropout_rates="0.5,17,0.05,78,0.9,5" \
  --data_dir=data/cifar10/0.8 \
  --train_log_dir=cifar_models/cifar10/resnet/0.8/mentornet_dd1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir=cifar_models/cifar10/resnet/0.8/mentornet_dd1/train \
  --eval_dir=cifar_models/cifar10/resnet/0.8/mentornet_dd1/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &

# CIFAR-100
python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/nl_0.2_percentile_90/mentornet_dd \
  --loss_p_precentile=0.8 \
  --fixed_epoch_after_burn_in \
  --burn_in_epoch=18 \
  --example_dropout_rates="0.5,17,0.05,78,0.9,5" \
  --data_dir=data/cifar100/0.2 \
  --train_log_dir=cifar_models/cifar100/resnet/0.2/mentornet_dd1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=0 >stdout0.txt 2>stderr0.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.2/mentornet_dd1/train \
  --eval_dir=cifar_models/cifar100/resnet/0.2/mentornet_dd1/eval_val \
  --studentnet=resnet101 --device_id=1 >stdout1.txt 2>stderr1.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/nl_0.2_percentile_90/mentornet_dd \
  --loss_p_precentile=0.6 \
  --fixed_epoch_after_burn_in \
  --burn_in_epoch=18 \
  --example_dropout_rates="0.5,17,0.05,78,0.9,5" \
  --data_dir=data/cifar100/0.4 \
  --train_log_dir=cifar_models/cifar100/resnet/0.4/mentornet_dd1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=2 >stdout2.txt 2>stderr2.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.4/mentornet_dd1/train \
  --eval_dir=cifar_models/cifar100/resnet/0.4/mentornet_dd1/eval_val \
  --studentnet=resnet101 --device_id=3 >stdout3.txt 2>stderr3.txt &

python code/cifar_train_mentornet.py --dataset_name=cifar100 \
  --trained_mentornet_dir=mentornet_models/models/nl_0.2_percentile_90/mentornet_dd \
  --loss_p_precentile=0.5 \
  --fixed_epoch_after_burn_in \
  --burn_in_epoch=18 \
  --example_dropout_rates="0.5,17,0.05,78,0.9,5" \
  --data_dir=data/cifar100/0.8 \
  --train_log_dir=cifar_models/cifar100/resnet/0.8/mentornet_dd1/train \
  --studentnet=resnet101 --max_number_of_steps=39000 --device_id=4 >stdout4.txt 2>stderr4.txt &

python code/cifar_eval.py --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir=cifar_models/cifar100/resnet/0.8/mentornet_dd1/train \
  --eval_dir=cifar_models/cifar100/resnet/0.8/mentornet_dd1/eval_val \
  --studentnet=resnet101 --device_id=5 >stdout5.txt 2>stderr5.txt &
