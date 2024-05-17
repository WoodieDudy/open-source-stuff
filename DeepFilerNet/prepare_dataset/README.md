# How to Create a Dataset for Training DeepFilterNet3

[DeepFilterNet3 Paper](https://arxiv.org/pdf/2305.08227):
```
We train the slightly modified DeepFilterNet model on the full
multi-lingual DNS4 dataset [11], while oversampling the highquality PTDB and VCTK datasets by a factor of 10 and evaluate
on the unseen VCTK/DEMAND test set.
```

### Prerequisites
You need 2TB of free space.

### Download datasets
Download the raw DNS4 dataset using the provided [script](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh)  
PTDB dataset from there https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html  
VCTK/DEMAND already located in DNS4  

### Install Dependencies
```sh
apt install python3.10-venv python3-h5py libhdf5-dev zip git -y
python -m venv venv
source venv/bin/activate

# Install CPU/CUDA PyTorch (>=1.9)
pip install torch torchaudio h5py
# Install DeepFilterNet including data loading functionality for training (Linux only)
pip install "deepfilternet[train]"
```

Run the script to prepare the dataset:  
```sh
python prepare_dataset.py
```

### Create HDF5 Files
Clone the DeepFilterNet repository:
```sh
git clone https://github.com/Rikorose/DeepFilterNet.git
cp make_hdf5_files.sh changes.patch Path_to_DeepFilterNet/DeepFilterNet
cd Path_to_DeepFilterNet/DeepFilterNet
git apply changes.patch # Apply changes to handle errors in prepare_data.py
export PYTHONPATH=$PWD
```

Run the script to create HDF5 files:
```sh
# Run the script from the Path_to_DeepFilterNet/DeepFilterNet directory
bash make_hdf5_files.sh
```

The dataset files will now be located in `/datasets/DNS4/dns4_hdf5_for_training`.

### Training the Model
Copy the configuration file:
```sh
cp dataset.cfg /datasets/DNS4/
```

Unzip the checkpoint:
```sh
unzip Path_to_DeepFilterNet/models/DeepFilterNet3.zip -d Path_to_DeepFilterNet/models/
```

Train the model:
```sh
# cd to Path_to_DeepFilterNet/DeepFilterNet
python df/train.py /datasets/DNS4/dataset.cfg /datasets/DNS4 DeepFilterNet/models/DeepFilterNet3
```
