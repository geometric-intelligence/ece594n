# Codes for ECE594N Project

## Preparation
Create the Environment
```
conda env create -f environment.yml
```

## Reproduce the Experimental Results
Results for different model architecures
```
python main_arch.py --save_path <YOUR_PATH_FOR_SAVING>
```
Results for different anomalous data
```
python main_data.py --save_path <YOUR_PATH_FOR_SAVING>
```

## Visualize the Results
Check vis.ipynb