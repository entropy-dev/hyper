# Step 01: Vis No Spectral Correction

```
roslaunch ximea preprocessing_standalone.launch
python Extraction.py -o /data/vis_data/2017_05_22/NoSpectralCorrection/PreprocessedImages/
python Playback.py /home/sflorian92/fusessh/

python Dataset\ Builder\ Generic.py -s /data/vis_data/2017_05_22/NoSpectralCorrection/PreprocessedImages/ -d /data/vis_data/2017_05_22/NoSpectralCorrection/HyperspectralDataVis20170522.hdf5 -i "256,512,16" -b 134217728
```

# Step 02 Vis With Spectral Correction

```
roslaunch ximea preprocessing_standalone.launch
python Extraction.py -o /data/vis_data/2017_05_22/WithSpectralCorrection/PreprocessedImages/
python Playback.py /home/sflorian92/fusessh/

python Dataset\ Builder\ Generic.py -s /data/vis_data/2017_05_22/WithSpectralCorrection/PreprocessedImages/ -d /data/vis_data/2017_05_22/WithSpectralCorrection/HyperspectralDataVis20170522.hdf5 -i "254,510,15" -b 134217728
```

# Step 03 Training

```
python TrainVis.py
```
