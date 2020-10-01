# Benchmark setup



<br>

## 1. Setup Conda

```
# Conda installation

# For Linux
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For OSX
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

chmod +x ~/miniconda.sh    
./miniconda.sh  

source ~/.bashrc          # For Linux
source ~/.bash_profile    # For OSX
```


<br>

## 2. Setup Python environment for CPU

```

# Clone GitHub repo
!git clone https://github.com/Saro00/DGN

# Install python environment
! conda env update --name base --file DGN/realworld_benchmark/environment_cpu.yml  --prune
```



<br>

## 3. Setup Python environment for GPU

DGL requires CUDA **10.0**.

For Ubuntu **18.04**

```
# Setup CUDA 10.0 on Ubuntu 18.04
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb 
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y cuda-10-0
sudo reboot
cat /usr/local/cuda/version.txt # Check CUDA version is 10.0

# Clone GitHub repo
!git clone https://github.com/Saro00/DGN

# Install python environment
! conda env update --name base --file DGN/realworld_benchmark/environment_gpu.yml  --prune
```



For Ubuntu **16.04**

```
# Setup CUDA 10.0 on Ubuntu 16.04
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y cuda-10-0
sudo reboot
cat /usr/local/cuda/version.txt # Check CUDA version is 10.0

# Clone GitHub repo
!git clone https://github.com/Saro00/DGN

# Install python environment
! conda env update --name base --file DGN/realworld_benchmark/environment_gpu.yml  --prune
```
<br>

## 4. Setup Colab

```
# Install miniconda
! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
! PATH=${PATH}:/usr/local/lib/python3.7/site-packages/

# Clone GitHub repo
!git clone https://github.com/Saro00/DGN

# Install python environment
! conda env update --name base --file DGN/realworld_benchmark/environment_gpu.yml  --prune
```


## 5. Download Datasets

```
# At the root of the repo
cd realworld_benchmark/data/ 
bash download_datasets.sh
```
