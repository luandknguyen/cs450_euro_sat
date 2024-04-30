# Project Introduction
Our team looks to identify land use via satellite imaging through the use of computer vision AI models. These have practical and beneficial uses in fields such as city planning or environmental monitoring as it allows you to classify large land-masses accurately and quickly.

Using the EuroSAT dataset, we looked to improve upon the AI classification model that a previous research group had used to identify land use using satelite imaging [EuroSAT repository](https://github.com/phelber/eurosat). This team used a classifier model (2-layer CNN) using Sentinel-2 satellite images w/ a 12-spectral band dataset. Our team looks to improve upon this model by using a U-Net network to predict the class of each pixel within the RGB dataset. 

# EuroSAT Image Segmentation

This repo defines a U-Net model for image segmentation of the EuroSAT dataset. The EuroSAT is a dataset of 27,000 geo images. Each image is 64-by-64 and colored.

The `models.py` script contains the definitions for the U-Net model and 2-layer CNN classifier.

The `Train.ipynb` notebook is used to train both models.

The `ModelAnalysis.ipynb` notebook is used to showcase the trained model.

# Dependencies

Required dependencies:
- Pytorch (CUDA)
- Pytorch Vision
- Numpy
- Scipy
- Matplotlib
- Scikit Image
- TQDM

Note: this repo assumes the user has CUDA support pytorch.

Installation:

```
    pip install tqdm scikit-image matplotlib numpy scipy
```

To install CUDA-supported pytorch, refer to [Pytorch](https://pytorch.org/get-started/locally/).

# Train

1. Download the RGB dataset from the [EuroSAT repository](https://github.com/phelber/eurosat).
2. Unzip it into the workspace root directory.
3. Renaming it into `dataset`.
4. If you make a new file and label it `dataset`, ensure contents mimic the original `EuroSAT` folder (has folders with classification of images)
5. Run the `Train.ipynb` notebook.

# Example

Example input image:

![Input Image](outputs/Sample%202.png)

The image is classified as "Residential".

### Class Probabilities Prediction

![Prediction](outputs/Sample%202%20Grid.png)

### Segmented Image

![Segmented Image](outputs/Sample%202%20Segmented.png)

The model marked the blue region as residential, the red region as highway, and the yellow region as river.

# Resources

[EuraSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/eurosat)

Helber, Patrick, et al. “Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification.” IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, July 2018, https://doi.org/10.1109/igarss.2018.8519248. 

Helber, Patrick, et al. “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification.” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, July 2019, pp. 2217–2226, https://doi.org/10.1109/jstars.2019.2918242. 
