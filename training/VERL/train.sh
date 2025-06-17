#!/bin/bash
#SBATCH --job-name="sql_grpo_v1.0.1"
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0-00:00:00
#SBATCH -p cu_0001
  
DOCKER_IMAGE=docker://your.registry.com/your_image:your_tag
SHARED_STORAGE="/your_shared_storage"
SHARED_STORAGE_DATA="/your_shared_storage/data"

# Set Docker credentials in environment variables
export DOCKER_LOGIN="your_username"
export DOCKER_PASSWORD="your_password" # Consider using a more secure method to store passwords
  
# Authenticate with Docker registry
echo "$DOCKER_PASSWORD" | docker login your.registry.com --username "$DOCKER_LOGIN" --password-stdin
  
# Check if Docker login was successful
if [ $? -ne 0 ]; then
    echo "Docker login failed"
    exit 1
fi
  
# CUDA variables
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
  
# Set SSL certificate path
export SSL_CERT_FILE=/path/to/your/certificates/roots.pem
# Runtime variables
export OMPI_MCA_coll_hcoll_enable=0
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="Your_Project_Name"
export WANDB_CACHE_DIR="$SHARED_STORAGE_DATA/wandb/cache"
export WANDB_CONFIG_DIR="$SHARED_STORAGE_DATA/wandb/config"
export WANDB_DIR="$SHARED_STORAGE_DATA/wandb"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_ETAG_TIMEOUT=500
  
# NCCL variables
export NCCL_DEBUG=DEBUG
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_BUFFSIZE=2097152
export NCCL_NVLS_ENABLE=0
  
# veRL variables
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export TRAIN_FILES=$SHARED_STORAGE_DATA/train_file.parquet
export TEST_FILES=$SHARED_STORAGE_DATA/test_file.parquet
  
export TRAINER_ARGS="\
custom_reward_function.path="/path/to/your/reward.py" \
custom_reward_function.name=your_function_name \
algorithm.adv_estimator=grpo \
data.train_files=\"$TRAIN_FILES\" \
data.val_files=\"$TEST_FILES\" \
data.train_batch_size=2048 \
data.max_prompt_length=1024 \
data.max_response_length=2048 \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=your_model_path \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.use_kl_in_reward=False \
trainer.critic_warmup=0 \
trainer.logger=['console','wandb'] \
trainer.project_name=$WANDB_PROJECT \
trainer.experiment_name='experiment_name' \
trainer.n_gpus_per_node=8 \
trainer.nnodes=$SLURM_NNODES \
trainer.save_freq=1 \
trainer.test_freq=5 \
trainer.total_epochs=20 $@
"

# Get master address and port
export MASTER_PORT=6379
export MASTER_NAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NAME" hostname --ip-address)
  
# Install on all nodes
srun -l -w $SLURM_NODELIST --kill-on-bad-exit=1 \
--no-container-mount-home \
--container-image=$DOCKER_IMAGE \
--container-name=$SLURM_JOB_NAME \
--container-mounts=$SHARED_STORAGE:$SHARED_STORAGE,./:/workspace \
--container-workdir=/workspace \
/bin/bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate py_3.11
pip install -q --no-deps --force-reinstall git+https://github.com/your_project.git@your_commit_hash
pip install -q -r requirements.txt --index-url https://pypi.org/simple/
"
  
# Start Ray head node
srun --nodes=1 --ntasks=1 -l -w "$MASTER_NAME" \
--no-container-mount-home \
--container-image=$DOCKER_IMAGE \
--container-name=$SLURM_JOB_NAME \
--container-mounts=$SHARED_STORAGE:$SHARED_STORAGE,./:/workspace \
--container-workdir=/workspace \
/bin/bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate py_3.11
ray start --head --node-ip-address=$MASTER_ADDR --port=$MASTER_PORT \
--dashboard-host=0.0.0.0 --dashboard-port=8265 \
--num-gpus=${SLURM_GPUS_PER_TASK} \
--node-name ${MASTER_NAME} \
--block
" &

sleep 10

echo "Submitting training job"
srun --overlap --nodes=1 --ntasks=1 -l -w "$MASTER_NAME" \
--no-container-mount-home \
--container-image=$DOCKER_IMAGE \
--container-name=$SLURM_JOB_NAME \
--container-workdir=/workspace \
/bin/bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate py_3.11
ray status --address=$MASTER_ADDR:$MASTER_PORT
ray job submit --address=http://$MASTER_ADDR:8265 -- python -m verl.trainer.main_ppo $TRAINER_ARGS $@
"