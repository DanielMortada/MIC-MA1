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

The custom CNN achieved a test accuracy of 80.67%, compared to the ResNet-18 baseline's 100%. Training took approximately 28 minutes and 47 seconds, reaching its best validation performance of 83.33% at epoch 47. The model shows steady improvement throughout training, with validation accuracy increasing from 5.33% at epoch 1 to over 80% in later epochs.

**Training Progress Highlights:**

* **Early Training (Epochs 1-10):** Rapid initial learning with validation accuracy improving from 5.33% to 34.67%
* **Mid Training (Epochs 11-30):** Continued improvement with validation accuracy reaching 66.00%
* **Late Training (Epochs 31-47):** Fine-tuning of features with validation accuracy peaking at 83.33%

The difference between validation (83.33%) and test (80.67%) accuracy indicates good generalization without significant overfitting.

**Insights:**

* Building a CNN from scratch allows for greater control and understanding of the model's inner workings.
* The custom architecture demonstrates strong learning capability, achieving over 80% accuracy on a 30-class problem without transfer learning.
* The performance gap compared to the baseline (19.33% lower than ResNet-18) highlights the benefits of transfer learning for complex tasks.
* The steady improvement over 47 epochs shows the model's ability to gradually build a hierarchical representation of musical instruments.

### Comparison with ResNet-18 Baseline

| Model | Parameters | Test Accuracy | Training Time | Best Epoch | Input Size |
|-------|------------|---------------|---------------|------------|------------|
| ResNet-18 (Transfer Learning) | 11.7 million | 100% | ~11m 20s | 8 | 224x224 |
| Custom CNN (From Scratch) | 8.6 million | 80.67% | ~29m | 47 | 224x224 |

Key differences:

* The custom model achieved respectable accuracy without any pre-training
* ResNet-18 converged much faster (best performance at epoch 8 vs. epoch 47)
* The custom model is more parameter-efficient (8.6M vs 11.7M parameters)

### Areas of Potential Improvement

Several avenues for improvement can be explored:

* **Architecture Optimization:** Experimenting with different filter sizes, layer depths, and block configurations.
* **Regularization Techniques:** Adjusting dropout rates, exploring L2 regularization, and data augmentation specific to musical instruments.
* **Training Strategy:** Implementing learning rate schedules, gradual unfreezing, or curriculum learning approaches.
* **Ensemble Methods:** Combining multiple models for increased robustness and performance.
* **Limited Transfer Learning:** Exploring the middle ground by initializing only early layers with pre-trained weights.
