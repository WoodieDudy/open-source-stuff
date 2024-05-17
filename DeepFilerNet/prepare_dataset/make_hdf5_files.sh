mkdir /datasets/DNS4/dns4_hdf5_for_training/

echo "Making VCTK_TRAIN_SET_SPEECH.hdf5"
python df/scripts/prepare_data.py --sr 48000 speech /datasets/DNS4/audio_paths/vctk_wav48_silence_trimmed_training_set_speech.txt /datasets/DNS4/dns4_hdf5_for_training/VCTK_TRAIN_SET_SPEECH.hdf5
echo "Making PTDB_speech_TRAIN.hdf5"
python df/scripts/prepare_data.py --sr 48000 speech /datasets/DNS4/audio_paths/ptdb_training_set_speech.txt /datasets/DNS4/dns4_hdf5_for_training/PTDB_speech_TRAIN.hdf5

echo "Making TRAIN_SET_SPEECH.hdf5"
python df/scripts/prepare_data.py --sr 48000 speech /datasets/DNS4/audio_paths/training_set_speech.txt /datasets/DNS4/dns4_hdf5_for_training/TRAIN_SET_SPEECH.hdf5
echo "Making TRAIN_SET_NOISE.hdf5"
python df/scripts/prepare_data.py --sr 48000 noise /datasets/DNS4/audio_paths/training_set_noise.txt /datasets/DNS4/dns4_hdf5_for_training/TRAIN_SET_NOISE.hdf5
echo "Making TRAIN_SET_RIR.hdf5"
python df/scripts/prepare_data.py --sr 48000 rir /datasets/DNS4/audio_paths/training_set_rirs.txt /datasets/DNS4/dns4_hdf5_for_training/TRAIN_SET_RIR.hdf5

echo "Making VAL_SET_SPEECH.hdf5"
python df/scripts/prepare_data.py --sr 48000 speech /datasets/DNS4/audio_paths/validation_set_speech.txt /datasets/DNS4/dns4_hdf5_for_training/VAL_SET_SPEECH.hdf5
echo "Making VAL_SET_NOISE.hdf5"
python df/scripts/prepare_data.py --sr 48000 noise /datasets/DNS4/audio_paths/validation_set_noise.txt /datasets/DNS4/dns4_hdf5_for_training/VAL_SET_NOISE.hdf5
echo "Making VAL_SET_RIR.hdf5"
python df/scripts/prepare_data.py --sr 48000 rir /datasets/DNS4/audio_paths/validation_set_rirs.txt /datasets/DNS4/dns4_hdf5_for_training/VAL_SET_RIR.hdf5
 
echo "Making TEST_SET_SPEECH.hdf5"
python df/scripts/prepare_data.py --sr 48000 speech /datasets/DNS4/audio_paths/test_set_speech.txt /datasets/DNS4/dns4_hdf5_for_training/TEST_SET_SPEECH.hdf5
echo "Making TEST_SET_NOISE.hdf5"
python df/scripts/prepare_data.py --sr 48000 noise /datasets/DNS4/audio_paths/test_set_noise.txt /datasets/DNS4/dns4_hdf5_for_training/TEST_SET_NOISE.hdf5
echo "Making TEST_SET_RIR.hdf5"
python df/scripts/prepare_data.py --sr 48000 rir /datasets/DNS4/audio_paths/test_set_rirs.txt /datasets/DNS4/dns4_hdf5_for_training/TEST_SET_RIR.hdf5

echo "Done!"
