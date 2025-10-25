\# Question 4 – Image Classification Using GP-based Feature Extraction



\## Requirements



To run the code for Question 4, you need to install the following Python packages:



```bash

pip install numpy pandas matplotlib pillow scikit-learn scipy deap scikit-image

```



These packages cover all dependencies used in the scripts, including:



\* `pandas`

\* `numpy`

\* `matplotlib.pylab`

\* `Pillow (PIL)`

\* `scikit-learn` (`LinearSVC`, `train\_test\_split`, `accuracy\_score`, `cross\_val\_score`, etc.)

\* `scipy.signal`

\* `deap` (for genetic programming)

\* `scikit-image` (`hog`, `local\_binary\_pattern`)



> Note: Some internal modules (like `evalGP\_main`, `sift\_features`, `feature\_function`, etc.) are part of the project files and are not installable via pip.



---



\## Required Project Files



The following Python scripts must be present in the working directory to execute the code:



\* `dataset\_reader\_example\_code.py`

\* `evalGP\_main.py`

\* `feature\_extractors.py`

\* `feature\_function.py`

\* `gp\_restrict.py`

\* `IDGP\_main.py`

\* `sift\_features.py`

\* `strongGPDataType.py`



Additionally, the following scripts are used to run the tasks for Question 4:



\* `run\_q41.py` – Runs GP-based feature extraction and evolves the best feature extractors.

\* `run\_q42.py` – Trains an image classifier (e.g., Linear SVM) using the features extracted by GP and evaluates its performance.



---



\## Data Requirements



All relevant data files must be placed in a `data` folder, including:



\* Training and test images (organized in subfolders for each class).

\* Preprocessed `.npy` files for train/test data and labels (if already generated).



The code assumes that the folder structure matches the format used in `dataset\_reader\_example\_code.py`.



---



\## How to Run



1\. \*\*Step 1 – Feature Extraction (Problem 4.1):\*\*



&nbsp;  ```bash

&nbsp;  python run\_q41.py

&nbsp;  ```



&nbsp;  This generates:



&nbsp;  \* `f1\_train\_patterns.csv`, `f1\_test\_patterns.csv`

&nbsp;  \* `f2\_train\_patterns.csv`, `f2\_test\_patterns.csv`

&nbsp;  \* `f1\_best\_tree.txt`, `f2\_best\_tree.txt`



2\. \*\*Step 2 – Image Classification (Problem 4.2):\*\*



&nbsp;  ```bash

&nbsp;  python run\_q42.py

&nbsp;  ```



&nbsp;  This trains a Linear SVM classifier on the extracted features and prints training and test accuracy.



---



This README ensures anyone can replicate your Question 4 experiments by installing the required packages, having all the scripts and data files in place, and running the provided scripts in order.



---







