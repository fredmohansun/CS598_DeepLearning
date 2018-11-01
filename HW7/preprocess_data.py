import numpy as np
import os
import nltk
import itertools
import io

## loading Imdb data

if not os.path.isdir('preprocessed_data'):
    os.mkdir('preprocessed_data')

work_dir = '/projects/training/bauh/NLP/aclImdb/'

train_pos_filenames = [work_dir+'train/pos/'+file for file in os.listdir(work_dir + 'train/pos/')]
train_neg_filenames = [work_dir+'train/neg/'+file for file in os.listdir(work_dir + 'train/neg/')]
train_unsup_filenames = [work_dir+'train/unsup/'+file for file in os.listdir(work_dir + 'train/unsup/')]

train_filenames = train_pos_filenames + train_neg_filenames + train_unsup_filenames

print(len(train_filenames))

train_inputs = []

for i, file in enumerate(train_filenames):
    with io.open(file,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    train_inputs.append(line)

test_pos_filenames = [work_dir+'test/pos/'+file for file in os.listdir(work_dir + 'test/pos/')]
test_neg_filenames = [work_dir+'test/neg/'+file for file in os.listdir(work_dir + 'test/neg/')]

test_filenames = test_pos_filenames + test_neg_filenames

print(len(test_filenames))

test_inputs = []
for i, file in enumerate(test_filenames):
    with io.open(file,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    test_inputs.append(line)

## number of tokens per review
no_of_tokens = []
for tokens in train_inputs:
    no_of_tokens.append(len(tokens))
no_of_tokens = np.array(no_of_tokens)
print(np.sum(no_of_tokens), np.min(no_of_tokens), np.max(no_of_tokens), np.mean(no_of_tokens),np.std(no_of_tokens))

## word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(train_inputs)
word_to_id = {token: i for i, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(train_inputs)
id_to_word = [token for _, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

print(id_to_word.shape)
## let's sort the indices by word frequency instead of random
train_inputs_token_ids = [[word_to_id[token] for token in x] for x in train_inputs]
count = np.zeros(id_to_word.shape)
for x in train_inputs_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
    print(id_to_word[i],count[i])

## recreate word_to_id based on sorted list
word_to_id = {token: i for i, token in enumerate(id_to_word)}

## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
train_inputs_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in train_inputs]
test_inputs_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in test_inputs]

## save dictionary
np.save('preprocessed_data/imdb_dictionary.npy',np.array(id_to_word))

## save training data to single text file
with io.open('preprocessed_data/imdb_train.txt','w',encoding='utf-8') as f:
    for tokens in train_inputs_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

## save test data to single text file
with io.open('preprocessed_data/imdb_test.txt','w',encoding='utf-8') as f:
    for tokens in test_inputs_token_ids:
        for token in tokens:
             f.write("%i " % token)
        f.write("\n")

glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'
with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in train_inputs]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in test_inputs]

np.save('preprocessed_data/glove_dictionary.npy',glove_dictionary)
np.save('preprocessed_data/glove_embeddings.npy',glove_embeddings)

with io.open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

with io.open('preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
