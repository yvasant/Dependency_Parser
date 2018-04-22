from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

ar_tags = ['<ROOT>', 'name', 'nsubjpass', 'dobj', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse','det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj','list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp',"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

tag_vecs = []
for i in ar_tags:
    random_vec = np.random.uniform(-0.5,0.5,50)
    tag_vecs.append(random_vec)

model_path = 'models/50d.word2vec_vectors'
wsm = word2vec.Word2Vec.load(model_path)
vocab = list(wsm.wv.vocab)
vocab = set(vocab)
NULL = np.random.uniform(-0.5,0.5,50)
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors_50d.txt", binary = False)
modelVocab = list(glove_model.wv.vocab)
print("Glove Word2Vec model loaded...")
glove_modelVocab = list(glove_model.wv.vocab)
glove_modelVocab = set(glove_modelVocab)

def hidden_function(w, t, l):
    # Get feature representation of elements
    word = embedd(w,'w')
    tag = embedd(t,'b')
    label = embedd(l,'b')
    word = np.concatenate([word,tag,label],axis = 0)
    return word

def embedd(set, type='w'):
    # Represent element as a d-dimensional vector
    global NULL,tag_vecs,ar_tags      
    
    feature = []
    feature = np.asarray(feature)
    if type == 'b':
        for elements in set:
            if elements == "NULL":
                feature = np.concatenate([feature , NULL],axis = 0)
                continue
            elif elements in ar_tags:
                feature = np.concatenate([feature , np.asarray(tag_vecs[ar_tags.index(elements)])],axis = 0)
            else :
                feature = np.concatenate([feature ,np.random.uniform(-0.5,0.5,50)],axis = 0)
        
        
    elif type == 'w':
        for element in set:
            if element == "NULL":
                feature = np.concatenate([feature , NULL],axis = 0)
                
            elif element in vocab:
                feature = np.concatenate([feature , wsm[element]],axis = 0)
            elif element in glove_modelVocab:
                feature = np.concatenate([feature , glove_model[element]],axis = 0)
            else:
                # https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ
                feature_size = 50
                random_feature = np.random.uniform(-0.5,0.5,feature_size)
                # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
                feature = np.concatenate([feature , random_feature],axis = 0)
    else:   
        feature = 0

    return feature

def get_children(parent, pdt, buffer, type='b'):
    leftmost = 'NULL'
    rightmost = 'NULL'
    left_pos = 'NULL'
    right_pos = 'NULL'

    if parent in pdt:
        left_pos = pdt.index(parent)
        right_pos = len(pdt) - pdt[::-1].index(parent) - 1

        if left_pos != parent:
            leftmost = buffer[left_pos]
        else:
            left_pos = 'NULL'
        # TODO: If rightmost == leftmost , should i handle it this way?
        if right_pos != parent and right_pos != leftmost:
            rightmost = buffer[right_pos]
        else:
            right_pos = 'NULL'

    if type == 'b':
        return leftmost, left_pos, rightmost, right_pos
    elif type == 'l':
        return leftmost, left_pos
    elif type == 'r':
        return rightmost, right_pos

def top_three(buffer, stack, buffer_pos, tags):
    Sw, St = [], []

    # Get the top 3 words on stack, if less then 3 exist, add Null instead
    if len(stack) < 3:
        Sw = [buffer[elem] for elem in stack]
        St = [tags[elem] for elem in stack]
        while len(Sw) < 3:
            Sw.append('NULL')
            St.append('NULL')
    else:
        Sw = [buffer[elem] for elem in stack[:-4:-1]]  # "Pop" 3 elements from stack
        St = [tags[elem] for elem in stack[:-4:-1]]

    # Get the top 3 words on the buffer
    for i in range(0,3):
        if buffer_pos+i < len(buffer)-1:
            Sw.append(buffer[buffer_pos+i])
            St.append(tags[buffer_pos+i])
        else:
            Sw.append('NULL')
            St.append('NULL')

    return Sw, St

def first_two_children(x, buffer, pdt, tags, arc_tags):
    # There probably is a nicer way of finding the two first occurances...
    Sw, St, Sl = [], [], []
    for elem in x:
        i = 0
        for word in buffer:
            if elem == buffer.index(word) and i < 2:
                i += 1
                left_child, pos_l, right_child, pos_r = get_children(elem, pdt, buffer, 'b')
                Sw.extend([left_child, right_child])

                if pos_l == 'NULL':
                    St.append(pos_l)
                    Sl.append(pos_l)
                else:
                    St.append(tags[pos_l])
                    Sl.append(arc_tags[pos_l])

                if pos_r == 'NULL':
                    St.append('NULL')
                    Sl.append('NULL')
                else:
                    St.append(tags[pos_r])
                    Sl.append(arc_tags[pos_r])

        for missing in range(i*2,4):
            Sw.append('NULL')
            St.append('NULL')
            Sl.append('NULL')

    return Sw, St, Sl

def leftmost_children(x, buffer, pdt, tags, arc_tags):
    Sw, St, Sl = [], [], []

    for elem in x:
        left_child, pos_l, right_child, pos_r = get_children(elem, pdt, buffer, 'b')
        left_child_child, pos_l = get_children(left_child, pdt, buffer, 'l')
        right_child_child, pos_l = get_children(elem, pdt, buffer, 'r')

        Sw.extend([left_child_child, right_child_child])

        if pos_l == 'NULL':
            St.append(pos_l)
            Sl.append(pos_l)
        else:
            St.append(tags[pos_l])
            Sl.append(arc_tags[pos_l])

        if pos_r == 'NULL':
            St.append('NULL')
            Sl.append('NULL')
        else:
            St.append(tags[pos_r])
            Sl.append(arc_tags[pos_r])

    if len(x) == 1:  # Ugly quick fix
        Sw.extend(['NULL']*2)
        St.extend(['NULL']*2)
        Sl.extend(['NULL']*2)

    return Sw, St, Sl


def create_sets(buffer, stack, pdt, buffer_pos, tags, arc_tags):
    Sw, St, Sl = [], [], []
    # TODO: This can be done so much fancier, ugly code
    # PDT: Predicted Dependency Tree

    # Step 1 - Top three elements from stack and buffer

    Sw, St = top_three(buffer, stack, buffer_pos, tags)

    # Step 2 - Get the first and second leftmost / rightmost children of the top two words on the stack

    x = stack[-2:]

    if not x: # Ugly quick fix
        Sw.extend(['NULL']*12)
        St.extend(['NULL']*12)
        Sl.extend(['NULL']*12)

    temp_w, temp_t, temp_l = first_two_children(x, buffer, pdt, tags, arc_tags)
    Sw.extend(temp_w)
    St.extend(temp_t)
    Sl.extend(temp_l)

    if len(x) == 1:  # Ugly quick fix
        Sw.extend(['NULL']*4)
        St.extend(['NULL']*4)
        Sl.extend(['NULL']*4)

    # Step 3 - Get the leftmost of leftmost / rightmost of right- most children of the top two words on the stack

    temp_w, temp_t, temp_l = leftmost_children(x, buffer, pdt, tags, arc_tags)
    Sw.extend(temp_w)
    St.extend(temp_t)
    Sl.extend(temp_l)

    return Sw, St, Sl

def predict(buffer, stack, pdt, buffer_pos, tags, arc_tags):
    #print("I am here ")
    #print(arc_tags)
    Sw, St, Sl = create_sets(buffer, stack, pdt, buffer_pos, tags, arc_tags)
    return hidden_function(Sw,St,Sl)
    #h = self.hidden_function(Sw, St, Sl)
    #print(Sw, '\n', St, '\n', Sl, '\n')
