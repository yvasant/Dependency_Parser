from gensim.models import word2vec
import numpy as np
import nn
from sklearn.neural_network import MLPClassifier
# parser is here only 
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25,10), random_state=1)
y_train = []
y_test = []
x_train = []
x_test = []
#x_train = np.asarray(x_train)
moves = {0:"SH", 1:"LA", 2:"RA"}

#def __init__(model_path):
#    self.nn = nn.NN(model_path)  # TODO: Create arc tagger

def valid_move(i, stack, pred_tree):
    valid_move = []
    # There are elements left in the buffer.
    if(i < len(pred_tree)):
        valid_move.append(0)
    # There are more than two elements + root element in the stack
    if(len(stack)>=2):
        valid_move.append(1)
        valid_move.append(2)
    # There are only 1 element + the root element in the stack
    elif(len(stack)==2):
        valid_move.append(2)

    return valid_move


def move(i, stack, pred_tree, move):
    i = i
    stack = stack
    pred_tree=pred_tree
    # shift
    if(move == 0):
        stack.append(i)
        i+=1
    # left
    elif(move == 1):
        ind = stack.pop(-2)
        pred_tree[ind] = stack[-1]
    # right
    elif(move == 2):
        ind = stack.pop(-1)
        pred_tree[ind] = stack[-1]

    return i, stack, pred_tree


def update(words, gold_tags, gold_arclabels, gold_tree,i):
    global y_train,x_test,y_test
    buffer = words
    stack = []
    dummy = []
    dummy = np.asarray(dummy)
    pdt = [0]*len(words)
    #tags = self.tagger.update(words, gold_tags)
    #arc_tags = self.arc_tagger.update(words, gold_arclabels)  # TODO: This is wrong, currently only looking at words , not pdt and words
    x=0
    while (True):
        dummy = []
        g_move = gold_move(x,stack,pdt,gold_tree)
        if g_move is None:
            break
        #feature = features(words,tags,x,stack,pdt)
        if i==0:
            y_train.append(g_move)
        else:
            y_test.append(g_move)
        
        dummy = nn.predict(buffer, stack, pdt, x, gold_tags, gold_arclabels)
        if i==0:
            x_train.append(dummy)
        else:
            x_test.append(dummy)
        x, stack, pdt = move(x,stack,pdt,g_move)

    return pdt


def gold_move(i, stack, pred_tree, gold_tree):
    pdt = pred_tree
    gold_tree=gold_tree
    valid_moves = valid_move(i,stack,pdt)
    try:
        if(len(stack)>=2):
            stack_top = stack[-1]
            stack_sec = stack[-2]
            if(1 in valid_moves and gold_tree[stack_sec] == stack_top):
                heads = [x for x,word in enumerate(gold_tree) if word == stack_sec]
                valid=True
                for i in heads:
                    if(pdt[i]!=stack_sec):
                        valid = False
                if(valid):
                    return 1
            if(2 in valid_moves and gold_tree[stack_top] == stack_sec):
                heads = [x for x,word in enumerate(gold_tree) if word == stack_top]
                valid=True
                for i in heads:
                    if(pdt[i]!=stack_top):
                        valid = False
                if(valid):
                    return 2
        if(0 in valid_moves and i < len(pred_tree)):
            return 0
        else:
            return None
    except IndexError:
        return None


# datareader is here



def conllu(fp):
    returnList = []
    for line in fp.readlines():
        if(line[0] == "#"):
            continue
        wordList = line.split()
        returnList.append(wordList)

        if not wordList:
            temp_return = returnList
            returnList=[]
            yield temp_return

def trees(fp):
     
    for tree in conllu(fp):
        pos = list()
        word = list()
        label = list()
        Head = list()
        bigList = list()

        word.append('<ROOT>')
        pos.append('<ROOT>')
        label.append('<ROOT>')
        Head.append(0)
        
        if len(tree) > 1:
            for tokens in tree:
                if len(tokens)>0:
                    word.append(tokens[1])
                    pos.append(tokens[3])
                    label.append(tokens[7])
                    if tokens[6]=='_':
                        continue
                    Head.append(int(tokens[6]))
    
            bigList.append(word)
            bigList.append(pos)
            bigList.append(label)
            bigList.append(Head)

        else:
            bigList.append(["<End>"])
            bigList.append(["<End>"])
            bigList.append(["<End>"])
            bigList.append(0)

        yield bigList



def evaluate(train_file, test_file):
    n_examples = 3000 # Set to None to train on all examples
    global x_train,y_train
    with open(train_file,encoding="utf-8") as fp:
        for i, (words, gold_tags, gold_arclabels, gold_tree) in enumerate(trees(fp)):
            if(words[0] == "<ROOT>"):   
                update(words, gold_tags, gold_arclabels, gold_tree,0)
                #print("\rUpdated with sentence #{}".format(i))
                if n_examples and i >= n_examples:
                    print("Finished training data generation......")
                    break
            else:  
                print("Finished training data generation.......")
                break
    # par.finalize()

    # acc_k = acc_n = 0
    # uas_k = uas_n = 0
    print("training neural network begins .........")
    clf.fit(x_train, y_train) 
    print("training finished ..........")
    x_train = []
    y_train = []
    n_examples = None
    with open(test_file,encoding="utf-8") as fp:
        for i, (words, gold_tags, gold_arclabels, gold_tree) in enumerate(trees(fp)):
            if(words[0] == "<ROOT>"):   
                update(words, gold_tags, gold_arclabels, gold_tree,1)
                #print("\rUpdated with sentence #{}".format(i))
                if n_examples and i >= n_examples:
                    print("Finished testing data generation..........")
                    break
            else:
                print("Finished testing data generation...........")
                break
     


def main():
    
    #en_tags = ["<ROOT>""]    

    en_train_file = "en_ewt-ud-train.conllu"
    en_test_file = "en_ewt-ud-test.conllu"

    #model_path = 'models/word2vec_vectors'
    
    evaluate(en_train_file, en_test_file)

    p=clf.predict(x_test)
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == p[i]:
            count = count + 1 
    accu = count/len(y_test)
    accu = accu*100
    print("Accuracy with feed forward Neural : " "%.4f" % accu)

    #dataReader.evaluate(sv_train_file, sv_test_file, myParser)


if __name__ == '__main__':
    main()


