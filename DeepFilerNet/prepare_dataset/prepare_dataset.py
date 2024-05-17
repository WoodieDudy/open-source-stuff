import os
import glob
import shutil
import zipfile
import tarfile
import urllib.request

target_dns4_folder = "/datasets/DNS4"
target_vctk_testset_folder = "/datasets/VCTK_DEMAND_testset"
target_ptdb = "/datasets/PTDB"

dataset_root = "/datasets"
audio_paths_dir = "./dns4_splits_audio_paths"

if not os.path.exists(target_dns4_folder):
    raise FileNotFoundError(f"{target_dns4_folder} does not exist")
if not os.path.exists(target_vctk_testset_folder):
    raise FileNotFoundError(f"{target_vctk_testset_folder} does not exist")
if not os.path.exists(target_ptdb):
    raise FileNotFoundError(f"{target_ptdb} does not exist")

os.makedirs(dataset_root, exist_ok=True)
os.makedirs(audio_paths_dir, exist_ok=True)

urls = [
    "https://github.com/Rikorose/DeepFilterNet/files/13514163/test_set_noise.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514165/test_set_speech.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514168/test_set_rirs.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514147/training_set_speech03.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514148/training_set_speech02.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514151/training_set_speech01.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514155/training_set_speech00.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514162/training_set_noise.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514161/training_set_rirs.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514156/validation_set_speech.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514159/validation_set_rirs.txt",
    "https://github.com/Rikorose/DeepFilterNet/files/13514160/validation_set_noise.txt"
]

for url in urls:
    filename = os.path.join(audio_paths_dir, os.path.basename(url))
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            raise RuntimeError(f"Error downloading {url}: {e}")

training_files = [
    "training_set_speech03.txt",
    "training_set_speech02.txt",
    "training_set_speech01.txt",
    "training_set_speech00.txt"
]

# Combine training files into one
with open(os.path.join(audio_paths_dir, "training_set_speech.txt"), 'w') as outfile:
    for filename in training_files:
        with open(os.path.join(audio_paths_dir, filename), 'r') as infile:
            outfile.writelines(infile.readlines())
        os.remove(os.path.join(audio_paths_dir, filename))

# Function to process and filter paths in split files
def process_files(file_path, exclusion_strings):
    valid_paths = []
    count_excluded = 0
    with open(file_path, 'r') as file:
        for line in file:
            if "datasets_fullband" in line:
                processed_path = line[line.find("datasets_fullband"):].strip()
            else:
                processed_path = line.strip()

            processed_path = os.path.join(target_dns4_folder, processed_path)

            if not any(exclusion in processed_path for exclusion in exclusion_strings):
                valid_paths.append(processed_path)
            else:
                count_excluded += 1
    print(f"{count_excluded} files were excluded from {file_path}")
    return valid_paths

# Check which speakers are in the test set to remove them from the training set
test_files = set(os.path.splitext(file)[0] for file in os.listdir(os.path.join(target_vctk_testset_folder, "noisy_testset_wav")))
excluded_speakers = set(file_name.split("_", 1)[0] for file_name in test_files)
print(f"Speakers to exclude: {excluded_speakers}")

for filename in os.listdir(audio_paths_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(audio_paths_dir, filename)
        filtered_paths = process_files(file_path, excluded_speakers)
        with open(file_path, 'w') as file:
            file.writelines(f"{path}\n" for path in filtered_paths)

# Move all VCTK samples from training_set_speech.txt to a separate file
training_set_speech_file = os.path.join(audio_paths_dir, "training_set_speech.txt")
vctk_file = os.path.join(audio_paths_dir, "vctk_wav48_silence_trimmed_training_set_speech.txt")

with open(training_set_speech_file, 'r') as file:
    lines = file.readlines()

moved_lines_count = 0
with open(training_set_speech_file, 'w') as f1, open(vctk_file, 'w') as f2:
    for line in lines:
        if "vctk_wav48_silence_trimmed" in line:
            f2.write(line)
            moved_lines_count += 1
        else:
            f1.write(line)
print(f"{moved_lines_count} lines moved to {vctk_file}")

# Create a file with paths for PTDB
ptdb_file = os.path.join(audio_paths_dir, 'ptdb_training_set_speech.txt')
with open(ptdb_file, 'w') as file:
    for gender in ['MALE', 'FEMALE']:
        mic_path = os.path.join(target_ptdb, gender, 'MIC')
        for subdir, dirs, files in os.walk(mic_path):
            for filename in files:
                if filename.endswith('.wav'):
                    full_file_path = os.path.join(subdir, filename)
                    file.write(full_file_path + '\n')

# Check for intersections in splits

def read_file_paths(file_paths):
    all_paths = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            all_paths.update(file.read().splitlines())
    return all_paths

def print_intersections(intersection_set, set_names):
    if intersection_set:
        print(f"Intersection found between {set_names}: {intersection_set}")
    else:
        print(f"No intersection between {set_names}.")

categories = ["speech", "noise", "rirs"]
splits = ["training", "validation", "test"]
files_dict = {
    split: {
        category: glob.glob(os.path.join(audio_paths_dir, f"*{split}_set_{category}.txt"))
        for category in categories
    } for split in splits
}

print(files_dict)

data_files = {
    split: {
        category: read_file_paths(file_paths)
        for category, file_paths in split_files.items()
    } for split, split_files in files_dict.items()
}

for category in categories:
    training_validation_intersection = data_files["training"][category].intersection(data_files["validation"][category])
    print_intersections(training_validation_intersection, f"Training and Validation {category} sets")

    training_test_intersection = data_files["training"][category].intersection(data_files["test"][category])
    print_intersections(training_test_intersection, f"Training and Test {category} sets")

    validation_test_intersection = data_files["validation"][category].intersection(data_files["test"][category])
    print_intersections(validation_test_intersection, f"Validation and Test {category} sets")

shutil.copy("dataset.cfg", target_dns4_folder)
shutil.copytree(audio_paths_dir, os.path.join(target_dns4_folder, "audio_paths"))
print("Done!")
