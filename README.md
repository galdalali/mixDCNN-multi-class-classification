This is a instance of multiClass and multiLabel classification in caffe using MixDCNN.
Thanks to fine-grained image classification network by ZongYuan Ge - https://github.com/zongyuange/MixDCNN
Thanks to the tools of multi-label data conversion tool in https://github.com/ChenJoya/Caffe_MultiLabel_Classification by ChenJoya.

This is an instructions to train MixDCNN on the data from:
MAFAT Challenge - Fine-Grained Classification of Objects from Aerial Imagery
https://competitions.codalab.org/competitions/19854

# Steps

## 1. Manufacture data

First we need to crop the data and divide it into classes using crop_tiles_train.py script that will create the train and validation tiles from the full images and a .txt file for each tile that represent his classes/labeles in the next form:

```
.
.
MAFAT/crops/train/img/225036.png 0 2 1 0 1 1 1 0 1 1 1 1 1 1 1
MAFAT/crops/train/img/255036.png 0 5 5 1 1 1 1 1 1 1 1 1 0 1 1
MAFAT/crops/train/img/285036.png 1 3 2 1 1 0 1 1 0 1 1 1 1 1 1
.
.
```

Where each label represent:

```
### 1. Class 
  small vehicle,
  large vehicle
### 2. SubClass 
  minibus,
  hatchback,
  sedan,
  bus,
  minivan,
  truck,
  van,
  jeep,
  cement mixer,
  dedicated agricultural vehicle,
  tanker,
  crane truck,
  pickup,
  light truck,
  prime mover
### 3. Color
  red,
  black,
  blue,
  silver/grey,
  white,
  other,
  yellow,
  green,
### 4. sunroof
  sunroof,
  no_sunroof
### 5. luggage_carrier
  luggage_carrier,
  no_luggage_carrier
### 6. sunroof
  open_cargo_area,
  no_open_cargo_area
### 7. enclosed_cab
  enclosed_cab,
  no_enclosed_cab
### 8. 0.spare_wheel
  spare_wheel,
  no_spare_wheel
### 9. wrecked
  wrecked,
  no_wrecked
### 10. flatbed 
  flatbed,
  no_flatbed
### 11. ladder
  ladder,
  no_ladder
### 12. enclosed_box
  enclosed_box,
  no_enclosed_box
### 13. soft_shell_box 
  soft_shell_box,
  no_soft_shell_box
### 14. harnessed_to_a_cart 
  harnessed_to_a_cart,
  no_harnessed_to_a_cart
### 15. ac_vents
  ac_vents,
  no_ac_vents
```

### Note:
Data augmnation is a big  deal in the chanllnge of fine-grained image classification so here we are randomly manipulte every tile in terms of crop size, noise and rotation.

## 2. Manufacture lmdb data-sets

First you need to insert "convert_multilabel.cpp" to caffe/tools/ and then recompile caffe 

Then try to manufacture your own lmdb by the example command lines:

///// train //////
```
GLOG_logtostderr=1 ./build/tools/convert_multilabel --resize_height=227 --resize_width=227 --shuffle ~/caffe/models/mixDCNN/ /media/gal/USB/MAFAT/crops/train/label/train.txt ~/caffe-/models/mixDCNN/TrainImage ~/caffe/models/mixDCNN/TrainLabel 15
```

////// val ///////
```
GLOG_logtostderr=1 ./build/tools/convert_multilabel --resize_height=227 --resize_width=227 --shuffle ~/caffe/models/mixDCNN/ /media/gal/USB/MAFAT/crops/val/label/val.txt ~/caffe/models/mixDCNN/ValImage ~/caffe/models/mixDCNN/ValLabel 15
```

## 3. Compute image mean

///// train //////
GLOG_logtostderr=1 caffe/build/tools/compute_image_mean ~/caffe/models/mixDCNN/TrainImage TrainImage.binaryproto

////// val ///////
GLOG_logtostderr=1 caffe/build/tools/compute_image_mean ~/caffe/models/mixDCNN/ValImage ValImage.binaryproto

## 4. Finetune birdsnap

The caffemodel weights for the best performing models can be downloaded from the links below:

0. [MixDCNN-6xGoogleNet for BirdSnap](https://cloudstor.aarnet.edu.au/plus/index.php/s/GBU2lheAlUY8bCm/download)
0. [MixDCNN-4xGoogleNet for CLEF-Flower](https://cloudstor.aarnet.edu.au/plus/index.php/s/uVftj1Xg12h0AgY/download)
0. [MixDCNN-4xGoogleNet for CUB2011](https://cloudstor.aarnet.edu.au/plus/index.php/s/zuSOuC7ZiZy3yTn/download)

```
./build/tools/caffe train -solver models/mixDCNN/GoogleNet_solver.prototxt -weights models/mixDCNN/GoogleNet_birdsnap_6.caffemodel -gpu 0
```

### Disclaimer 
0. While make LMDB for training and testing set, make sure resize then to 227 by 227 to match the trained parameters.
0. We have tested the model parameters with caffe version 1.0.
0. To re-train or fine-tuning the models with our prototxt files, you need a decent GPU with 12GB memory (K40,K80,Titan X).

## 5. Classify test data
Put "classificationMulti.cpp" in /home/gal/caffe-1.0/examples/cpp_classification and re-make caffe

```
./build/examples/cpp_classification/classificationMulti.bin models/mixDCNN/MixDCNNMulti_deploy.prototxt models/mixDCNN/snapshot/MixDCNNMulti_iter_90000.caffemodel /media/gal/MyPassport/TrainImage/TrainImage.binaryproto models/mixDCNN/labels/labels_ac_vents.txt models/mixDCNN/labels/labels_class.txt models/mixDCNN/labels/labels_color.txt models/mixDCNN/labels/labels_enclosed_box.txt models/mixDCNN/labels/labels_enclosed_cab.txt models/mixDCNN/labels/labels_flatbed.txt models/mixDCNN/labels/labels_harnessed_to_a_cart.txt models/mixDCNN/labels/labels_ladder.txt models/mixDCNN/labels/labels_luggage_carrier.txt models/mixDCNN/labels/labels_open_cargo_area.txt models/mixDCNN/labels/labels_soft_shell_box.txt models/mixDCNN/labels/labels_spare_wheel.txt models/mixDCNN/labels/labels_subclass.txt models/mixDCNN/labels/labels_sunroof.txt models/mixDCNN/labels/labels_wrecked.txt MAFAT/crops/test\ crops/
```
