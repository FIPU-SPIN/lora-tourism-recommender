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

echo ">>> torch (CUDA 12.1 wheel)"
pip install \
    torch==2.5.1 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

echo ">>> transformers stack"
pip install \
    transformers \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    sentencepiece \
    sentence-transformers \
    faiss-cpu \
    tqdm

echo ""
echo "Verifying..."
python3 -c "
import torch, transformers, accelerate, bitsandbytes, huggingface_hub
import sentence_transformers, faiss, tqdm
print('torch               :', torch.__version__)
print('transformers        :', transformers.__version__)
print('accelerate          :', accelerate.__version__)
print('bitsandbytes        :', bitsandbytes.__version__)
print('huggingface_hub     :', huggingface_hub.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('tqdm                :', tqdm.__version__)
print('CUDA available      :', torch.cuda.is_available())
"
echo "Done. You can now run: qsub eval-tourism.pbs"
