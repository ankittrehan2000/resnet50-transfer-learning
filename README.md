# resnet50-transfer-learning
A sample model for Spotted Lantern Fly images that leverages transfer learning using the pre-trained ResNet50 model 

### Epoch training output
<img src="https://github.com/ankittrehan2000/resnet50-transfer-learning/blob/main/demo/resnetmodel.png" height="400" />

## Other info

The training script adds a few `Dense` layers on top of the ResNet50 model that comes out of the box with keras. This is done in order to train the model to cater to the needs of the specific types of image being passed to it while leverages the pre-trained weights from a massive dataset used to train the ResNet50 model.

The directory also has a `streamlit` web application that creates a web wrapper around the trained model which is exported by the training script, pre-processes the images before passing them to the model for classification and gives an output to show if the image has a Spotted Lantern Fly egg or not. 

Youtube Link: https://youtu.be/xBvgw46ht3Y
