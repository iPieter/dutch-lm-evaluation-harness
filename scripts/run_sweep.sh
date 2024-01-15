#!/bin/bash
#SBATCH --job-name=lm_eval_job           # Job name
#SBATCH --output=lm_eval_job_%j.out      # Output file
#SBATCH --error=lm_eval_job_%j.err       # Error file
#SBATCH --clusters=wice                  # Cluster
#SBATCH --partition=gpu                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=18             # Number of tasks (processes) per node
#SBATCH --gpus-per-node=1                # Number of tasks (processes) per node
#SBATCH --time=3:00:00                   # Walltime limit (hh:mm:ss)

# Set HF_HOME if VSC_SCRATCH_SITE doesn't exist
if [ -z "$VSC_SCRATCH_SITE" ]; then
  export HF_HOME=/cw/dtaijupiter/NoCsBack/dtai/pieterd/hf_cache
fi

venv_path=".env"

# Check if .env folder already exists
if [ ! -d "$venv_path" ]; then
  # .env folder does not exist, create and activate a new virtual environment
  conda activate py310-base
  python3 -m venv "$venv_path"
  source "$venv_path/bin/activate"

  # Install Python packages from requirements.txt
  pip install -r requirements.txt
else
  # .env folder already exists, activate the existing virtual environment
  source "$venv_path/bin/activate"
fi


# List of models
models=("FremyCompany/t2t-temp-model-2")
#models=("yhavinga/gpt-neo-1.3B-dutch" "Rijgersberg/GEITje-7B" "Rijgersberg/GEITje-7B-chat-v2" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mistral-7B-v0.1" "tiiuae/falcon-7b-instruct" "FremyCompany/t2t-temp-model-2")

# List of shots
shots=(0 1 2 5)  # Adjust the list of shots as needed

# Loop over models and shots
for model in "${models[@]}"; do
  for num_shots in "${shots[@]}"; do
    srun python lm_eval \
      --model hf \
      --model_args pretrained="$model" \
      --tasks squadnl \
      --device cuda:1 \
      --batch_size 8 \
      --num_fewshot "$num_shots" \
      --wandb_project dutch-lm-eval
  done
done

deactivate