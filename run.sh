TASK_NAME='cifar10'
exp_name=cifar10
gpu=1
seed=0

model_dir=../model/${exp_name}
log_dir=../log/${exp_name}_log.txt

label_ratio=0.05
alpha=0.5
num_users=50
rounds=500
sample_ratio=0.4
per_device_train_batch_size=32
entropy_threshold=0.1
ema_weight=0.9



CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=${gpu} \
python run.py \
--task_name ${TASK_NAME} \
--per_device_train_batch_size ${per_device_train_batch_size} \
--learning_rate_sup 0.01 \
--learning_rate_unsup 0.01 \
--seed ${seed} --debug False \
--entropy_threshold ${entropy_threshold} \
--label_ratio ${label_ratio} \
--alpha ${alpha} \
--log_dir ${log_dir} \
--model_dir ${model_dir} \
--num_users ${num_users} \
--rounds ${rounds} \
--sample_ratio ${sample_ratio} \
--ema_weight ${ema_weight} 