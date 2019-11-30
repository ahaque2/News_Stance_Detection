# News_Stance_Detection
Stance Detection in News articles using only textual features.

To run: 
python3 run.py

Recommended for faster execution: Download pre-generated word-embeddings and features, link: https://drive.google.com/drive/folders/1GqfriXYoMk15j12cSfXQ6Vmc-pZYgIBU?usp=sharing

To try out different approaches change folowing values:

To run with pre-generated word embeddingss (Google News): set 'pretrained_flag' = 1 (line 133) and 'word2vec_flag' = 1 (line 134) in run.py
To run with Tf-Idf and using a saved model: set 'pretrained_flag' = 1 (line 133) and 'word2vec_flag' = 0 (line 134) in run.py
To generate word-embeddings and use them (): set 'pretrained_flag' = 0 (line 133) and 'word2vec_flag' = 0 (line 134) in run.py
To use pre-generated word embedding (with no data-preprocessing): choose (uncomment) 'data_source' = "word_embedding_with_no_preprocessing/" in line(107) and comment out all other 'data_source initialization' (line 106 and 108)
