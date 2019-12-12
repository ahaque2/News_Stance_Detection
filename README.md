# News_Stance_Detection
Stance Detection in News articles using textual features.

### Pre-requisites **(You need to download the following and place (unzipped) under the root directory)**

1. Download Pre-trained Google News word2vec model from here [Pre-trained Google News model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and place it(unzipped bin file) in root directory
2. Download pre-generated feature vectors for different approaches from here [download_pregenerated_feature_vector_zip](https://drive.google.com/file/d/1nkfF5YYVV7EkxeufnaVg4qX-O91Dx_pW/view?usp=sharing) and place (unzipped) under the root directory.

We highly recommend to use the pre-generated feature vectors (mentioned in step 2 above) as generating these features may take a long time. If you still want to generate these files you can change the default config to generate it, steps are given later in this document.

The code includes multiple approaches to address the News Stance Detection problem. Follow the steps below to change configurations to run different approaches.

### How to run?

[approach_1.py](approach_1.py) includes the baseline model (By default uses pre-generated word-embeddings (recommended). To change to generate word-embeddings go to line *3145 in approach_1.py and change to 'pretrained_flag = 0')

[approach_2_and_3.py](approach_2_and_3.py): By default the code is configured to run the *Multi-Layer Perceptron* (MLP) model with pre-trained Google News word2Vec word-embeddings. To run different approaches you just need to change some flags in [approach_2_and_3.py](approach_2_and_3.py) file as following:

[Approach_4.ipynb](Approach_4.ipynb) and [Approach_5.ipynb](Approach_5.ipynb) : Load the python notebooks into google colab, switch to gpu model. Create a folder called 'Stance_detection' in your google drive and put the data folder inside that. Also upload the Pre-trained Google News word2vec model inside the 'Stance_detection' folder in google drive. Use the google colab to run the two approches.

#### Using pre-generated word-vectors (*recommended*) 
(Make sure you have [pre-generated_vectors](https://drive.google.com/file/d/1nkfF5YYVV7EkxeufnaVg4qX-O91Dx_pW/view?usp=sharing) zip folder (unzipped) under the same directory where [approach_2_and_3.py](approach_2_and_3.py) is)
There are 3 flags in [approach_2_and_3.py](approach_2_and_3.py) that you can use to change configurations, these flags are ***'pretrained_flag', 'word2vec_flag'***, and ***'data_source_id'***, at the lines ***153, 154*** and ***155*** respectively

- **data_source_id = 1**, pretrained_flag = 1, word2vec_flag = 1	(Default) (Uses pre-generated word-embeddings)
- **data_source_id = 2**, pretrained_flag = 1, word2vec_flag = 1	(Uses summarized text sample dataset pre-generated vectors)
- **data_source_id = 3**, pretrained_flag = 1, word2vec_flag = 1	(Uses word-embedding with no text-preprocessing)


#### Generating word-embeddings## (not recommended, as this may take hours to complete execution)

- Change to **pretrained_flag = 0** and **word2vec_flag = 1** to generate word-embeddings using Google News pretrained word2vec model as input for the model

Using Tf-Idf instead of word-embeddings:

- Change to **word2vec_flag = 0** to generate tf-idf and use as input features instead of word-embeddings
	


### Requirements:

The code is written in **python3**.
You need to have following libraries installed.

- tesorflow
- scikit learn
- gensim
- nltk
- numpy
- pandas
- scipy
- vaderSentiment
- Keras
- Tensorflow

In case you have any difficulty executing our code, or notice a missing file (or any other thing in general) feel free to contact either of us and we will be happy to help.
- Amanul Haque [ahaque2@ncsu.edu](ahaque2@ncsu.edu)
- Suvodeep Majumder [smajumd3@ncsu.edu](smajumd3@ncsu.edu)
