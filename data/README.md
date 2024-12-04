# download
follow the download guide for the kinetics-400 dataset on this [page](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md)

after the download is complete, please make the directory structure as follows
```
data/
├── Kinetics-400.tar.gz.0000
├── Kinetics-400.tar.gz.0001
├── preprocess.py
└── README.md
```

then, run the code to create a single tar.gz file
```
cat data/Kinetics-400.tar.gz.* > data/Kinetics-400.tar.gz
rm data/Kinetics-400.tar.gz.*
```

once the process is complete, the directory structure will be as follows
```
data/
├── Kinetics-400.tar.gz
├── preprocess.py
└── README.md
```