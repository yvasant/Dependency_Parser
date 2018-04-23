DataSet used : EWT (https://github.com/UniversalDependencies/UD_English-EWT/tree/dev)
Description of dataset : https://github.com/UniversalDependencies/UD_English-EWT/blob/master/README.md
 summary of dataset : 
The corpus comprises 254,830 words and 16,622 sentences, taken from five genres of web media: weblogs, newsgroups, emails, reviews, and Yahoo! answers. See the LDC2012T13 documentation for more details on the sources of the sentences. The trees were automatically converted into Stanford Dependencies and then hand-corrected to Universal Dependencies. All the basic dependency annotations have been single-annotated, a limited portion of them have been double-annotated, and subsequent correction has been done to improve consistency. Other aspects of the treebank, such as Universal POS, features and enhanced dependencies, has mainly been done automatically, with very limited hand-correction.
 
-There are two files run.py and nn.py in which run.py takes the .conllu file and generates states in the tree and which move to be followed and stores the move for training purpose and testing purpose. nn.py file takes stack, buffer, labels, pos tags and tree and returns the word embedding (feature vector) for that configuration.
- For training the model I am using feed forward neural network with two hidden layers with size of (25,10).
- I am only training first 3000 sentences because of large amount of memory consumption and testing on full test file given in the dataset.
- Accuracy: 90.87%