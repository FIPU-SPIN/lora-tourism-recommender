#!/bin/bash
# Run this ONCE on the login node to install all required packages into the venv.
# After this completes successfully, qsub eval-tourism.pbs will work without downloading anything.

set -e

cd "$(dirname "$0")"

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

module purge
module load utils/python/3.12.2
module load gcc/12.1.0

source venv/bin/activate

echo ">>> pip upgrade"
pip install --upgrade pip

echo ">>> install requirements"
pip install -r requirements.txt

echo ""
echo "Verifying..."
python3 -c "
import torch, transformers, accelerate, bitsandbytes, huggingface_hub
import peft, trl, datasets, pandas
import sentence_transformers, faiss, tqdm
print('torch               :', torch.__version__)
print('transformers        :', transformers.__version__)
print('accelerate          :', accelerate.__version__)
print('peft                :', peft.__version__)
print('trl                 :', trl.__version__)
print('datasets            :', datasets.__version__)
print('pandas              :', pandas.__version__)
print('bitsandbytes        :', bitsandbytes.__version__)
print('huggingface_hub     :', huggingface_hub.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('tqdm                :', tqdm.__version__)
print('CUDA available      :', torch.cuda.is_available())
"
echo "Done. You can now run: qsub eval-tourism.pbs"
