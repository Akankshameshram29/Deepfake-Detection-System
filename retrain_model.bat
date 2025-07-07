@echo off
REM Activate the conda environment and run retrain_model.py

CALL "C:\Users\akanksha meshram\anaconda3\Scripts\activate.bat" tf_env
python "C:\Users\akanksha meshram\OneDrive\Documents\Desktop\Deepfake\retrain_model.py"

@echo off
CALL "C:\Users\akanksha meshram\anaconda3\Scripts\activate.bat" tf_env
python "C:\Users\akanksha meshram\OneDrive\Documents\Desktop\Deepfake\retrain_model.py" >> retrain_log.txt 2>&1
