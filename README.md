# NN_CNN_Facial_Recognition

*Further details provided in the documentation within the multipurpose.py file.*

## Project
This is a emotion classifier using facial recognition technology. By using a pretrained Convolutoinal Nerual Network (The VGG16 model), imported from Keras, we retrained the final layer of the network on a dataset of 4900 images of a diverse group of individuals featuring 7 emotions (angry, surprised, sad, happy, afraid, disgusted, and neutral). Checkpoint where used not only to sample and make predictions through various points during training, but also to save progress and resume training when convenient. If you don't have a GPU, It is encouraged to train this model using Google Colab or a cloud service (Azure, GCS, AWS, etc.).

## Colaborators
Data preprocessing code was developed by @fzheng1 and was used to organize the dataset by directory to make for computational efficiency and remove the need for a direct corresponding labelling.

## Future plans and outcomes
The ultimate goal of the project is to be able to detect signs of depression through various external symptoms like sleep deprevation, malnurishment, and other factors. Emotional sentiment analysis is only one step of that process. Upwards and onwards, always and only :rocket:!
