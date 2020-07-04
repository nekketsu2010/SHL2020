REM python create_spectam.py Acc train 50 40 xy
REM python create_spectam.py Acc train 50 40
REM python create_spectam.py Gyr train 50 40 xy
REM python create_spectam.py Gyr train 50 40
REM python create_spectam.py Mag train 50 40 xy
REM python create_spectam.py Mag train 50 40
REM python create_spectam.py Acc validation 50 40 xy
REM python create_spectam.py Acc validation 50 40
REM python create_spectam.py Gyr validation 50 40 xy
REM python create_spectam.py Gyr validation 50 40
REM python create_spectam.py Mag validation 50 40 xy
REM python create_spectam.py Mag validation 50 40
REM python learn_spectram.py 50 40 1
REM python learn_spectram.py 50 40 2
REM python create_spectam.py Acc train 80 70 xy
REM python create_spectam.py Acc train 80 70
REM python create_spectam.py Gyr train 80 70 xy
REM python create_spectam.py Gyr train 80 70
REM python create_spectam.py Mag train 80 70 xy
REM python create_spectam.py Mag train 80 70
REM python create_spectam.py Acc validation 80 70 xy
REM python create_spectam.py Acc validation 80 70
REM python create_spectam.py Gyr validation 80 70 xy
REM python create_spectam.py Gyr validation 80 70
REM python create_spectam.py Mag validation 80 70 xy
REM python create_spectam.py Mag validation 80 70
REM python learn_spectram.py 80 70 1
REM python learn_spectram.py 80 70 2
REM python learn_spectram.py 64 8 1
REM python learn_spectram.py 64 8 2
REM python user_model.py 64 8 1 2
REM python user_model.py 64 8 1 3
REM python user_model.py 64 8 2 2
REM python user_model.py 64 8 2 3
REM python learn_spectram.py 64 8 1
REM python learn_spectram.py 64 8 2
REM python create_spectam.py Acc test 64 8 xy
REM python create_spectam.py Acc test 64 8
REM python create_spectam.py Gyr test 64 8 xy
REM python create_spectam.py Gyr test 64 8
REM python create_spectam.py Mag test 64 8 xy
REM python create_spectam.py Mag test 64 8
REM python create_spectam.py LAcc test 64 8 xy
REM python create_spectam.py LAcc test 64 8
REM python create_spectam.py Acc train 64 8 xy Bag
REM python create_spectam.py Acc train 64 8 Bag
REM python create_spectam.py Gyr train 64 8 xy Bag
REM python create_spectam.py Gyr train 64 8 Bag
REM python create_spectam.py Mag train 64 8 xy Bag
REM python create_spectam.py Mag train 64 8 Bag
REM python create_spectam.py LAcc train 64 8 xy Bag
REM python create_spectam.py LAcc train 64 8 Bag
REM python create_spectam.py LAcc validation 64 8 xy Bag
REM python create_spectam.py LAcc validation 64 8 Bag
REM python create_spectam.py Acc validation 64 8 xy Bag
REM python create_spectam.py Acc validation 64 8 Bag
REM python create_spectam.py Gyr validation 64 8 xy Bag
REM python create_spectam.py Gyr validation 64 8 Bag
REM python create_spectam.py Mag validation 64 8 xy Bag
REM python create_spectam.py Mag validation 64 8 Bag
REM python create_spectam.py Acc train 64 8 xy Torso
REM python create_spectam.py Acc train 64 8 Torso
REM python create_spectam.py Gyr train 64 8 xy Torso
REM python create_spectam.py Gyr train 64 8 Torso
REM python create_spectam.py Mag train 64 8 xy Torso
REM python create_spectam.py Mag train 64 8 Torso
REM python create_spectam.py LAcc train 64 8 xy Torso
REM python create_spectam.py LAcc train 64 8 Torso
REM python create_spectam.py LAcc validation 64 8 xy Torso
REM python create_spectam.py LAcc validation 64 8 Torso
REM python create_spectam.py Acc validation 64 8 xy Torso
REM python create_spectam.py Acc validation 64 8 Torso
REM python create_spectam.py Gyr validation 64 8 xy Torso
REM python create_spectam.py Gyr validation 64 8 Torso
REM python create_spectam.py Mag validation 64 8 xy Torso
REM python create_spectam.py Mag validation 64 8 Torso
REM python create_spectam.py Acc train 64 8 xy Hand
REM python create_spectam.py Acc train 64 8 Hand
REM python create_spectam.py Gyr train 64 8 xy Hand
REM python create_spectam.py Gyr train 64 8 Hand
REM python create_spectam.py Mag train 64 8 xy Hand
REM python create_spectam.py Mag train 64 8 Hand
REM python create_spectam.py LAcc train 64 8 xy Hand
REM python create_spectam.py LAcc train 64 8 Hand
REM python create_spectam.py LAcc validation 64 8 xy Hand
REM python create_spectam.py LAcc validation 64 8 Hand
REM python create_spectam.py Acc validation 64 8 xy Hand
REM python create_spectam.py Acc validation 64 8 Hand
REM python create_spectam.py Gyr validation 64 8 xy Hand
REM python create_spectam.py Gyr validation 64 8 Hand
REM python create_spectam.py Mag validation 64 8 xy Hand
REM python create_spectam.py Mag validation 64 8 Hand
python predict_spectram.py 64 8 train
python predict_spectram.py 64 8 validation
python predict_spectram.py 64 8 test
