# Handwritten Character Recognizer

![](/assets/main.jpeg)

In this project, I will demonstratesteps to train and deploy a Machine Learning model with a web interface. 

Tools required:
* A server for inference: Cloud instances, or simply your powerful laptop.
* Data: EMNIST Balanced an extended version of MNIST with 131,600 characters, 47 balanced classes.
* PyTorch: To train the deep learning model.
* Docker: Make our life easier to create a container for our application.
* Flask: For API and user interface.

### Workflow:
The steps are as follows:

 * Prepare the docker image using Dockerfile.
 * Get data and train the machine learning model on Colab.
 * Show a simple inference example.
 * Build the API and the user-interface

Now we need to think about how to “ship” our model. This what we call packaging. It is simply to put the inference steps together and have a simple API to use it later.

![](/assets/inference.png)