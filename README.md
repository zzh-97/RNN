# RNN
This repository provides RNN training script for cell classification in pathological images.

## RNN Training
Run ``` python RNN_train.py ``` to train the RNN model.  
* Script outputs
  * Train loss and train accuracy every 5 epoches.
  * Test accuracy

## Additional Information
The training data is loaded in the first part of this script

* The training data
  * Source: [Quantitative Biomedical Research Center of UT Southwestern Medical Center](https://qbrc.swmed.edu/projects/cnn/)
  * Type: Tiles extracted from a lung adenocarcinoma pathological image
  * Count: 12000 (4000 per class)
  * Size: 80 * 80 * 3
