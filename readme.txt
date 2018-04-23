The "Large Movie Review Dataset"(*) was used for this project. The dataset is compiled from a collection of 50,000 reviews from IMDB on the condition there are no more than 30 reviews per movie. The numbers of positive and negative reviews are equal. Negative reviews have scores less or equal than 4 out of 10 while a positive review have score greater or equal than 7 out of 10. Neutral reviews are not included. The 50,000 reviews are divided evenly into the training and test set.

The Training Dataset used is stored in the zipped folder: aclImbdb.tar file. This can also be downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz.

The Test Dataset is stored in the folder named 'test'

About files :
Bag of word representaion was given with the dataset. File BBOW.py contains code for getting Binary bag of word representation from given Bag of word representaion using regular expressions.

Similarly tf.py contains code for normalized term frequency representaion and tfidf.py contains code for term frequency inverse document frequency representation using tfidf.

And model_word2vec.py and model_Paragraph.py contains code for training word vectors and paragraph vectors respectively and saving the vectors.

Avg_Word2vec.py contains code for generating single vector for a document by taking average of vectors for all the words present in that document.

Avg_word2vec_with_TFIDF.py contains code for generating single vector for a document by taking average of vectors weighted with tfidf for all the words present in that document.

Main.py contains all of the above files combined so by running main you can get all of the four representaions(BBOW, tf, tfidf, word vectors, Paragraph vectors) for the dataset

All the files ending with _accuracy.py are containing the code for checking accuracy of that perticular representaion on four/three classifiers i.e. Logistic regression, Feed forward neural network, Support vector machine, Naive bayes in case of three Naive bayes is excluded. 

Results:
Maximum accuracy of 88.34% was achieved on word2vec representaion using Feed forward neural network with two hidden layers of 10 and 8 size.