# TumourVision G1 (TVG-1)

<!-- add more illustrations -->
<!-- make this a PyPI module -->
<!-- later add X accuracy -->
<!-- may require logistic regression analysis -->

TumourVision G1 (TVG-1) incorporates [PolyTrend](https://github.com/asiimwemmanuel/polytrend) methodologies to improve brain tumor detection in MRI scans. Utilizing a Convolutional Neural Network (CNN) architecture developed with PyTorch, TVG-1 classifies brain MRI scans for tumor detection. It employs Torch, TorchVision, and Matplotlib libraries for analysis and visualization. With roughly 23.64 million parameters, TVG-1 provides a robust computational framework. TVG-1 achieved a 92.16% accuracy on the test set. Considerations for TVG-2 are in place and in active development, enhancing diagnostic capabilities for medical professionals. The TumourVision generation 1 (TVG-1) model architecture is built around the popular ResNet50 model and applies transfer learning, with modified fully connected layers at the end following the convolutional layers.

> `CLOD_formula.png` in full is Convolutional Layers' Output Dimensions formula. Just as an abbreviation for easy file naming.

## Features

- Integration of PolyTrend's polynomial trend analysis for data preprocessing and feature extraction.
- Utilization of polynomial regression techniques to refine CNN predictions and improve classification accuracy.
- Seamless integration with existing TVG-1 architecture for easy adoption and deployment.

## Dataset

The dataset used for training and testing TVG-1 was sourced from two Kaggle datasets:

- [Brain Tumor Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Installation

To install and set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Josh-The-Developapa/TVG-1.git
   cd TVG-1
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

Now you should have all the necessary dependencies installed and be ready to run the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [**Joshua Mukisa**](https://github.com/josh-the-developapa) - Developer of **TVG-1**
- [**Emmanuel Asiimwe**](https://github.com/asiimwemmanuel/) - Developer of **PolyTrend**