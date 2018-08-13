# Indoor-Positioning

## MSc Project: Indoor localization using neural networks

### Brief introducton: ###
1. try to use clollected wifi fingerprint to train a rough neural network to roughly predict the location of a mobile device
2. using the accelerometer and the magnetometer data to enhence the time consistency of the location prediction(trajectory continuity)
### Current progress: ###
Current progress:###
- collected data insite, and get the ouput files from the prebuilt android mobile app(more information available:  https://github.com/vradu10/LSR_DataCollection.git). 
- preprocessed the data file, and converted them into standard inputs and outputs that the neural nets required.
- constructed 2 simple neural nets(classification, regression) to predict the location from wifi fingerprint
- implemented autoencoder layerwise to pretrain the neural nets(make use of the large amount of unlabeled wifi data collected previously)
- compare different network strcuctures(\[32,64,16\] and \[200,200,200\]). Meantime, see how dropout layer and autoencoder pretrained weights helps the prediction process.
- get the transition probability matrix, and the median wr 'matrix'(each element in this two matrices indicate transition between two grid\[start_grid -> row, end_grid -> column\]).
- implement the Hidden Markov Model to enforce time consistency(2 adjacent timestep's location do not differ too much -> tragectory continuity).

### Current results visualization: ### 

The following plots is the "error in meters cdf" of different models. More detailed plots(such as error line plot, training curve plot .etc) can be found in results(*) directory. Note: C indicates classification models, while R indicates regression models.

simple vs dropout:

![simple vs dropout](https://github.com/gracecxj/Indoor-Positioning/blob/master/comparison1/CDF1.png)

simple vs autoencoder:

![simple vs autoencoder](https://github.com/gracecxj/Indoor-Positioning/blob/master/comparison2/CDF2.png)

autoencoder vs autoencoder+dropout:

![autoencoder vs autoencoder+dropout](https://github.com/gracecxj/Indoor-Positioning/blob/master/comparison3/CDF3.png)


### To be continue:### 
- think about a way to integrate the accelerometer and magnetometer data to the inputs.
- collecting two dataset: dynamically and staticlly

### Code Documentation:### 

The main part of this repository are three directories:

**1. `"Data"` directory**
-  `"background"` directory : the original data collected by the mobile app
- `"masking area"` directory : the two types of masking areas (forground)
- `"unlabelled"` directory : the preprocessed unlabelled WiFi data (no location)

**2. `"Source code" `directory**
* `"preprocessing"` directory
    * `WifiPreprocessing.py` : scan all the wifi signal in background file, write distinct access point into a file, it can iterate over all the background files in a specified directory
    * `Masking.py` : the masking function defined in this file can take a list of positions as inputs, which construct a polygon area. index the whole world rectangular space(60*80), and then return the grid index which are inside the polygon.
    * `unlabelled.py` :  traverse all background files and get the unlabelled wifi data (used only for training autoencoders)
* `"utilities"` directory
    * `Plotting.py` : some functions related to plot the training curve and the cdf
    * `sdae.py` :  constructing a layer-wise training process of autoencoders
    * `SensorParse.py` : Pre-processing the background files, generate standard input data, and instantiate a SensorFile object for each background file collected
    * `SensorParse2.py` : some changes for hmm over the previous code
* `"core"` directory
    * `main.py` : training the models (classification and regression)
    * `main_sdae.py` : training the models (classification and regression) that are pretrained by autoencoders
    * `main_hmm.py` : training the models (classification and regression) that refined by HMM

**3. `"Experiement results"` directory**
- some output cdf plots and error line plots of different models


