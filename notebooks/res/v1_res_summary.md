## Musical Instrument Classification with a Custom CNN

This notebook explores musical instrument classification using a custom Convolutional Neural Network (CNN). A fine-tuned ResNet-18 model achieved 100% accuracy as a baseline, but the focus here is on developing a model from scratch to gain a deeper understanding of CNN architecture design and tailor a solution for the musical instrument dataset.

### Model Architecture

The custom CNN architecture follows a pyramidal structure with 5 convolutional blocks for progressive feature extraction:

* **Block 1:** 3 → 32 → 32 channels (basic edge and texture detection)
* **Block 2:** 32 → 64 → 64 channels (more complex patterns)
* **Block 3:** 64 → 128 → 128 channels (instrument parts and shapes)
* **Block 4:** 128 → 256 → 256 channels (high-level instrument features)
* **Block 5:** 256 → 512 → 512 channels (complex features, suitable for 224x224 input)

Each block consists of two convolutional layers with batch normalization and ReLU activation, followed by max pooling and dropout for regularization. Global average pooling reduces feature maps to a 512-dimensional vector, fed into a classifier head with a hidden layer, dropout, and a final output layer for 30 musical instrument classes.

### Key Design Principles

The architecture incorporates these key design principles:

* **Progressive Feature Extraction:** Increasing channel depth allows the network to learn a hierarchy of features.
* **Repeated Convolutional Blocks:** A balanced architecture for efficient feature learning.
* **Parameter Efficiency:** Small kernel sizes, batch normalization, and progressive dropout.
* **Global Average Pooling:** Reduces parameters and improves robustness.
* **Classifier Head Design:** Facilitates complex feature combinations and prevents overfitting.

### Results and Insights

The custom CNN achieved a test accuracy of 76%, compared to the ResNet-18 baseline's 100%. Training took approximately 1 hour and 14 minutes, reaching its best validation performance at epoch 12. Confusion matrices provide insights into class-wise performance and potential misclassifications.

**Insights:**

* Building a CNN from scratch allows for greater control and understanding of the model's inner workings.
* The custom architecture demonstrates the ability to learn relevant features for musical instrument classification.
* The performance gap compared to the baseline highlights the benefits of transfer learning for complex tasks.
* Architectural choices and regularization play a crucial role in model performance and convergence.

### Areas of Potential Improvement

Several avenues for improvement can be explored:

* **Architecture Optimization:** Experimenting with different filter sizes, layer depths, and block configurations.
* **Regularization Techniques:** Adjusting dropout rates, exploring L2 regularization, and data augmentation specific to musical instruments (e.g., audio-based augmentations).
* **Ensemble Methods:** Combining multiple models for increased robustness and performance.
* **Hyperparameter Tuning:** Fine-tuning learning rate, batch size, and optimizer parameters for optimal convergence.
