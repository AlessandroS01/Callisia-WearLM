# _Dataset description_

The PPG-Dalia dataset was converted in a format different from the original one. To obtain the version used by this 
application, run the following method from the ```PPGDaliaDatasetHandler``` class:

- ```extract_data``` specifying for each single subject the path to its original data. More precisely, remark the 
    large .pkl file.

By doing that, the dataset will be converted in a format that is more suitable for the application. In particular, the 
data will be split for each single patient in a folder containing:

- `metadata.json`: **file containing the metadata of the patient.**
- `activity.csv`: **file containing the activity labels.** 
  - _Sampled at 4Hz_
- `label.csv`: **includes the ground truth heart rate information.** 
  - _Mean of the ECG-based instantaneous heart rate_
  - _Obtained with sliding window of 8 seconds, shifted with 2 seconds_
- `rpeaks.csv`: **file containing the indexes of the identified and corrected R-peaks, referring to the ECG-signal.** 
  - _R-peaks provide the basis of the heart rate ground truth_
- `chest` folder for the RespiBAN data: 
  - `ACC.csv`: **Three-axis acceleration acquired via a 3D-accelerometer** 
    - _Sampled at 700Hz_
  - `ECG.csv`: **ECG data** 
    - _Sampled at 700Hz_
  - `Resp.csv`: **Respiration signal** 
    - _Sampled at 700Hz_
- `wrist` folder for the Empatica E4 data: 
  - `ACC.csv`: **Three-axis acceleration acquired via a 3D-accelerometer** 
    - _Sampled at 32Hz_
    - _Data provided in units of 1/64g_
  - `BVP.csv`: **Blood Volume Pulse (BVP) data**
    - _Sampled at 64Hz_
  - `EDA.csv`: **Electrodermal Activity (EDA) data**
    - _Sampled at 4Hz_
    - _Data provided in units of μS_
  - `TEMP.csv`: **Skin temperature data**
    - _Sampled at 4Hz_
    - _Data provided in units of °C_