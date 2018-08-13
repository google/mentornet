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

export PYTHONPATH="$PYTHONPATH:$PWD/code/"

# Pre-Defined
python code/training_mentornet/data_generator.py --outdir=mentornet_models/data/mentornet_pd_example --vstar_fn=vstar_mentornet_pd --vstar_gamma=1 --sample_size=100000
python code/training_mentornet/train.py --train_dir=mentornet_models/models/mentornet_pd_example --data_path=mentornet_models/data/mentornet_pd_example --max_step_train=3e4 --learning_rate=0.1

# Data-Driven
python code/training_mentornet/data_generator.py --vstar_fn=data_driven --input_csv_filename=mentornet_models/studentnet_dump_csv/nl_0.2_datapara_0.75_7k.csv --outdir=mentornet_models/data/mentornet_dd_example
python code/training_mentornet/train.py --train_dir=mentornet_models/models/mentornet_dd_example --data_path=mentornet_models/data/mentornet_dd_example/nl_0.2_datapara_0.75_7k_percentile_90 --max_step_train=3e4 --learning_rate=0.1

# Visualization
python code/training_mentornet/visualizer.py --model_dir=mentornet_models/models/mentornet_pd_example --loss_bound=5 --epoch_ranges="0,10,18,20,30,40,50,60,70,80,90,95,99"
python code/training_mentornet/visualizer.py --model_dir=mentornet_models/models/mentornet_dd_example --loss_bound=15 --epoch_ranges="0,10,18,20,30,40,50,60,70,80,90,95,99"
