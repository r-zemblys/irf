# IRF (Identification by Random Forest)
Eye-movement event detection using random forest. Cite as:
```sh
@article{zemblys2018irf,
  title={Using machine learning to detect events in eye-tracking data},
  author={Zemblys, Raimondas and Niehorster, Diederick C and Komogortsev, Oleg and Holmqvist, Kenneth},
  journal={Behavior research methods},
  volume={50},
  number={1},
  pages={160--181},
  year={2018},
}
```

## Running IRF code
IRF was developed using Python 2.7 programming language and number of packages for data manipulation and training machine learning algorithms. This section describes how to prepare required software, how to use IRF algorithm and how to train your own classifier.

### 1. Prepare python environment
An easy way of preparing your python environment is to use [Anaconda](https://www.anaconda.com/what-is-anaconda/) - an open source package management system and environment management system that runs on Windows, macOS and Linux. To install Anaconda follow the instructions provided in <https://www.anaconda.com/download/>, then open your terminal and type:

```sh
conda create --name irf python=2.7
source activate irf
```
The next step is to install all required python libraries. Run the following commands in your terminal window:

```sh
pip install tqdm
pip install parse
pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install astropy
pip install scikit-learn
```

Note that if you want to use a pretrained classifier that comes with this code, use:
```sh
pip install scikit-learn==0.17.1
```

To check if your environment is prepared correctly, run:
```Shell
python run_irf.py --help
```

You should see the following output:
```sh
usage: run_irf.py [-h] [--ext EXT] [--output_dir OUTPUT_DIR]
                  [--workers WORKERS] [--save_csv]
                  clf root dataset

Eye-movement event detection using Random Forest.

positional arguments:
  clf                   Classifier
  root                  The path containing eye-movement data
  dataset               The directory containing experiment data

optional arguments:
  -h, --help            show this help message and exit
  --ext EXT             File type
  --output_dir OUTPUT_DIR
                        The directory to save output
  --workers WORKERS     Number of workers to use
  --save_csv            Save output as csv file
```
  
### 2. Parse eye-movement data

This package includes a hand-labeled eye-movement dataset, called `lookAtPoint_EL` (see in `./etdata/`). To parse this data using IRF, [download](https://doi.org/10.5281/zenodo.1343920) a pretrained model, unzip and place it in `./models/` directory and run:
```Shell
python run_irf.py irf_2018-03-26_20-46-41 etdata lookAtPoint_EL
```

You can also use custom `--output_dir` parameter if you like, otherwise output folder will be set to `./etdata/lookAtPoint_EL_irf`. 

**!!!** Note that pretrained models were trained using this exact dataset. Also note that only approximate saccade and blink events were manually coded for trial `lookAtPoint_EL_S3` and it was not used for training or testing classifier.

After running the above command you will get messages like:

```sh
etdata/lookAtPoint_EL_irf/i2mc/lookAtPoint_EL_S1_i2mc.mat does not exist. Run i2mc extractor first!
etdata/lookAtPoint_EL_irf/i2mc/lookAtPoint_EL_S2_i2mc.mat does not exist. Run i2mc extractor first!
...
```
One of the features (i2mc) requires third party software. Running IRF for the first time converts data into the format, that is required for I2MC the algorithm. Open `./util_lib/I2MC-Dev/I2MC_rz.m` in MATLAB, edit `folders.data ` to point to your output directory and run the code. It will extract and save i2mc features. Note that I2MC code uses random initiations to calculate data clusters and therefore each time you recalculate i2mc feature, it will be slightly different. Therefore if you care about reproducing your classification, use the same already extracted i2mc data.

Now run `python run_irf.py irf_2018-03-26_20-46-41 etdata lookAtPoint_EL` again. IRF will parse your data and save it as structured numpy arrays. It has also an option to save output in tab delimited text format: just add parameter `--save_csv` when running IRF.

#### 2.1. Parsing your own data
The internal data format used by IRF is a structured numpy array with a following format:
```
dtype = np.dtype([
	('t', np.float64),	#time in seconds
	('x', np.float32),	#horizontal gaze direction in degrees
	('y', np.float32), 	#vertical gaze direction in degrees
	('status', np.bool),	#status flag. False means trackloss 
	('evt', np.uint8)	#event label:
					#0: Undefined
					#1: Fixation
					#2: Saccade
					#3: Post-saccadic oscillation
					#4: Smooth pursuit
					#5: Blink
])
```
That means one first needs to convert the dataset to this format. Note that dataset folder needs to have `db_config.json` file, that describes the geometry of the setup - physical screen dimensions in mm, eye distance in mm and screen resolution in pixels, for example:
```
    "geom": {
        "screen_width": 533.0,
        "screen_height": 301.0,
        "eye_distance": 565.0,
        "display_width_pix": 1920.0,
        "display_height_pix": 1080.0
    }
```
Geometry also needs to be defined in `./util_lib/I2MC-Dev/I2MC_rz.m`. **Note that dimensions here are in cm!** After preparing your data run the IRF code in a similar way described above. 

## Train your own classifier
### 1. Data
To train your own classifier place your training data into `dataset/train` and your validation data into `dataset/val` directories. Note that `dataset` directory needs to contain `db_config.json` file that describes the geometry of the setup. Training and validation data needs to be in the structured numpy array format described above.

You can use `./utils_lib/data_prep/augment.py` script to prepare `lookAtPoint_EL` dataset for training the IRF. Just run the script and it will augment data by resampling it to various sampling rates and will add noise to it. Furthermore the script will split data into the training/validation and testing sets. Remember to copy `db_config.json` to `lookAtPoint_EL/training/`.
**Note that `augment.py` was developed using an older version of numpy, therefore you might need to replace you numpy instalation with version 1.11 by running:**
```sh
pip install numpy==1.11
```

### 2. Training
In `config.json` you can adjust the training parameters:
```
{
    "events": [1, 2, 3],	#event labels to use; only fixation (1), saccade (2) and pso (3) are tested
    "n_trees": 32,		#number of trees to use
    "extr_kwargs": {		#feature extraction parameters
        "w": 100,		#context size for calculating features; in ms
        "w_vel": 12,			
        "w_dir": 22,
        "interp": false,	#not used
        "print_et": false	#not used
    },
    "features": [		#features to use
	"fs",
	"disp",
	"vel",
	"acc",
	"mean-diff",
	"med-diff",
	"rms",
	"std",
	"bcea",
	"rms-diff",
	"std-diff",
	"bcea-diff",
	"rayleightest",
	"i2mc"
    ]
}
```
Now run:
```
python run_training.py etdata/lookAtPoint_EL training
```
This will perform feature extraction, train the IRF classifier and save it to the `./models/irf_datetime` directory. Note that the training script will stop if the `i2mc` feature is used, in case of which you will need to run `./util_lib/I2MC-Dev/I2MC_rz.m` before actually training the classifier. After i2mc is extracted, rerun the training script one more time.
