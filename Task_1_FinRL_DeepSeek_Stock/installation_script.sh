mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

nano ~/.bashrc
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
conda --version


conda create --name myenv python=3.10 -y
conda init 
source ~/.bashrc
conda activate myenv

conda install -c conda-forge gcc -y
conda install -c conda-forge swig -y
conda install -c conda-forge box2d-py -y
conda install -c conda-forge mpi4py -y
conda install -c conda-forge datasets -y
conda install -c conda-forge huggingface_hub -y
conda install -c conda-forge alpaca-py -y
conda install -c conda-forge selenium -y
conda install -c conda-forge webdriver-manager -y
pip install git+https://github.com/benstaf/FinRL.git
git clone https://github.com/benstaf/spinningup_pytorch.git
cd spinningup_pytorch
pip install -e .
cd ..


git clone https://github.com/benstaf/FinRL_DeepSeek.git
cd FinRL_DeepSeek

mpirun --allow-run-as-root -np 8 python train_ppo_deepseek.py


