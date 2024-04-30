# README: 

The EuroSAT is a dataset for land use and classification that consists of 27,000 satellite images covering 10 different classes. The classes can be such as agriculture, industrial and more. 


To run the code you simply need to run each block of the program and when you finish running all the blocks the program will give you an estimate of how correct each part was classified by the program.


To install the dependencies you simply have to run the first block of code and that will be able to provide you all the data sets that you need and the libraries that you need for the code to run and test.


The dependencies that were used was PyTorch, NumPy, MatPlot.


PyTorch – used for the classification (CNN) and segmentation (U -Net) model for image classification. Implements loss functions. Used for data loaders and datasets for efficient preprocessing of the datasets. We use this to predict the pixels


NumPy – this provides mathematical functions for calculations. This is used to convert the images such as resizing and cropping. Used to calculate accuracy, precision base on model prediction.


MatPlot – this was used for sample images from different land cover classes. Used for the graphs which visualize the loss.
 
The code will get all image that is provided in the data set and the program will classify which part belongs. This is done through calculating rte proabability of each pixel and which class it would belong into. The output will be numbers that represent the probability of each pixel of each class.


## Resources:

Helber, Patrick, et al. “Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification.” IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, July 2018, https://doi.org/10.1109/igarss.2018.8519248. 

Helber, Patrick, et al. “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification.” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, July 2019, pp. 2217–2226, https://doi.org/10.1109/jstars.2019.2918242. 


Websites that use those 2 articles:

EuroSat Dataset (kaggle.com)

GitHub - phelber/EuroSAT: EuroSAT: Land Use and Land Cover Classification with Sentinel-2
