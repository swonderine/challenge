# Outlier Detection

A simply outlier detection in Python

## Getting Started

These instructions briefly tell you how to get this project running.

### Prerequisites

Project is created with Python 3.8.5.

#### Packages

* pandas: 1.2.3
* numpy: 1.20.1
* matplotlib: 3.3.4

Please make sure you have the above listed packages installed. Otherwise, install the packages according to your python3 environment, e.g:

```
$ pip3 install pandas numpy matplotlib
```

### Clone repository

Clone the repository to your desired local folder.

```
$ cd /path/to/yourlocalfolder
$ git clone https://github.com/swonderine/challenge.git

```
### Run script 

Before running the script make sure the file containing the data, [train.gz](https://www.kaggle.com/c/avazu-ctr-prediction/data) is copied into the same folder of the repository, where [challenge.py](https://github.com/swonderine/challenge/blob/main/challenge.py) sits. Then you can run the script via `python3`

```
$ cd challenge
$ cp /path/to/train.gz .
$ python3 challenge.py
```
The script saves two PNG images, showing the results of the two tasks.

### Output Images

The CTR over time is shown in `ctr_ts.png`.
The CTR, simple moving averages (MA_12h, MA_24h) over time and according outliers are shown in `ctr_ts_ma_outlier.png`. The outliers are calculated via different simple moving averages (MA_12h, MA_24h) and according standard deviations. Additionally, calculated outliers using the standard deviation of the full series are shown (+,x).
