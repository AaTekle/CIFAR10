# CIFAR-10 Image Classifier
Creator: Aaron Tekle
# Link: https://cifar10.streamlit.app/


## Project Overview

This project demonstrates the end-to-end development of a Convolutional Neural Network (CNN) for classifying images into one of the ten classes in the CIFAR-10 dataset. The relevant classes include airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships, and trucks.

## Model Development and Training

### Creation

This CNN model was designed with a focus on extracting and learning the most relevant features from the CIFAR-10 dataset's images. The architecture comprises several key components:

- **Convolutional Layers**: To automatically identify features within the images.
- **Pooling Layers**: To reduce the dimensionality of the feature maps, thus simplifying the network.
- **Dropout Layers**: To prevent overfitting by randomly dropping units from the neural network during training.
- **Dense Layers**: To classify the images based on the features extracted and pooled by the previous layers.

### Training

The training process involved the following steps:

1. **Data Preprocessing**: Images were normalized to ensure pixel values were between 0 and 1, making it easier for the model to learn.
2. **Augmentation**: To improve the robustness of the model, data augmentation techniques such as random flips and rotations were employed.
3. **Optimization**: The Adam optimizer was chosen for its efficiency in updating the network weights.
4. **Validation**: A portion of the dataset was reserved for validation to monitor the model's performance on unseen data.

After several iterations, the model reached an accuracy of approximately 76% on the training dataset, underscoring its ability to generalize well to new data.

## Web Application and Deployment

The trained model was deployed as an interactive web application using Streamlit. This allows users to upload images and receive predictions in real-time. The deployment process involved:

1. **Streamlit Setup**: The web app was scripted using Streamlit, highlighting its ease of transforming scripts into shareable apps.
2. **Model Integration**: The trained model was integrated into the app, enabling direct interaction with users.
3. **Deployment**: The Streamlit app was deployed, making it accessible to users worldwide to upload images and test the model's performance.

## Tools/Technologies used

- **Python** for programming the model and web application.
- **TensorFlow & Keras** for creating and training the CNN.
- **Streamlit** for deploying the model as an interactive web application.
- **PIL** (Python Imaging Library) for image processing tasks.

## License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation:
If using the CIFAR-10 dataset, please cite:
```
Krizhevsky, A., Nair, V., and Hinton, G. (2009). CIFAR-10 (Canadian Institute for Advanced Research).
```

## Extra Sources
Sources that helped me throughout this process:

- https://www.cs.toronto.edu/~kriz/cifar.html

- https://www.deeplearningbook.org/contents/convnets.html (This Helped a lot)

- https://learnopencv.com/understanding-convolutional-neural-networks-cnn/ (took the image off of this site)

## Acknowledgments
Special thanks to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for their work in creating the CIFAR-10 dataset.

- TensorFlow and Keras teams for their comprehensive libraries that facilitate model building and training.
