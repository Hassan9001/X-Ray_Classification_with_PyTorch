# X-Ray_Classification_with_PyTorch

Deep neural network model to classify the presense of pneumonia in a patient using their chest X-ray.

### Data Description

The dataset originates from Kermany et al. on Mendeley.
The particular subset used for this project is sourced via Kaggle <https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>.
    Acknowledgements
    Data: <https://data.mendeley.com/datasets/rscbjbr9sj/2>
    Citation: <http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5>
The sourcedata consisted of 300 training and 100 testing chest X-ray images. An additional validation split is created using 100 randomly selected images from the training set. Each dataset is evenly divided into two classes—NORMAL and PNEUMONIA—comprising X-ray images of healthy lungs and those affected by pneumonia.
Data Structure:
    - data/train: 200 images (100 NORMAL, 100 PNEUMONIA)
    - data/val: 100 images (50 NORMAL, 50 PNEUMONIA)
    - data/test: 100 images (50 NORMAL, 50 PNEUMONIA)
All images are preprocessed and resized to 224x224 pixels using transforms.Resize(224) and transforms.CenterCrop(224).
Data is loaded into a train_loader, val_loader, and test_loader using the DataLoader class from the PyTorch library.

### Methods

- Data Augmentation & Normalization: X-rays from the Training data are applied with transformations and normalization ('train_transform'), while Validation and Testing datasets are applied with only normalization, not augmentation ('val_test_transform').
    ○ Normalization is made consistent with the ResNet-50 input domain with a normalize function that takes as input the means and standard deviations of the three color channels, (R,G,B), from the original ResNet-50 training dataset
    ○ Random horizontal flipping is used to augment data and improve model robustness.
- Pre-Trained Model: Fine-tuned the ResNet-50 model (ResNet50_Weights.IMAGENET1K_V2), a convolutional neural network that was pre-trained on ImageNet before optimizing it on our data.
    ○ Using pre-defined weights allows for accurate classification, faster training time, and fewer resource costs.
- Transfer Learning & Fine-Tuning: Only the final layer of the network is retrained, reducing the computational cost and the amount of required training data.
    ○ Freezes the model’s convolutional layers and replaces the final fully connected layer with a single output neuron for binary classification (normal vs. pneumonia).
- Efficient Training Loop With Early Stopping: Includes a validation set to monitor model performance and trigger early stopping after a set number of epochs without any improvements to the model, preventing overfitting while saving on computational resources.
    ○ Reduces the learning rate when validation loss plateaus using ReduceLROnPlateau
    ○ Saved best model checkpoint
- Performance Metrics: Measures accuracy and F1-score on the test set to provide a comprehensive view of the model’s classification performance.
- Hyperparameters: Optionally can adjust the learning rate, batch size, and number of epochs in the script as desired.
- Model Architecture: Optionally can experiment with different pre-trained models (e.g., DenseNet, EfficientNet) by making minimal changes to the code.

### Results

    - Test Accuracy: ~0.810 (81.0%)
    - Test F1-Score: ~0.838 (83.8%)
These scores indicate the model performs well in distinguishing pneumonia-affected lungs from normal lungs.
