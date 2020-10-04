#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os


# In[21]:


BASE_DIR = os.getcwd()
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'debate')


# ### We assume that each line in the file is an element of the training data

# In[23]:


training_X = []

#read all files in "debate" folder one by one

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    fname = os.path.join(TEXT_DATA_DIR, name)
    
    #if data format ".txt" 
    
    if fname.endswith(".txt"):
        with open(fname) as infile:
            for line in infile:
                #clear possible spaces except carriage return and break
                line = line.strip()
                try:
                  training_X.append(line)
                except Exception as e:
                  print(e)


# In[25]:


#print the first and last lines
print (training_X[0])
print (training_X[-1])


# In[29]:


#print data count
len(training_X)


# ### Each row has been imported into the training_X list as an input data. Instead, all text in files can be added together. However, thanks to our operation, both text preprocessing and vector transformation can be progressed together.

# In[32]:


get_ipython().system('pip install keras')


# In[35]:


conda install tensorflow


# In[37]:


conda install keras


# In[38]:


from keras.preprocessing.text import Tokenizer
 
# create tokinizer object
 
# parameter options and default values
# num_words=None, lower=True, split=' ', char_level=False, oov_token=None
# filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ '
 
# Let's process the 10 most commonly used words in text
MAX_NUM_WORDS = 10
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

# Setting the tokenizer class according to the input data
tokenizer.fit_on_texts(training_X)


# ### At this time, the entry data is still stored in "training_X". But he knows a lot of things to know about tokinizer login data. These are: word list, word frequencies, most used 10 words

# In[40]:


#Let's take a closer look at what words have passed
for word in tokenizer.word_index:
    print (word)


# ### 2216 words that do not repeat each other were shown. However, this cannot be said to be perfect for some numbers. For example, 0,1,2,3,4,7,8,9 was shown twice. 

# In[43]:


print ("Summary ", len(tokenizer.word_index), " there are different words")
print ("Summary ", tokenizer.num_words, " pieces of word will be processed")


# ### All words are found according to - the split =  '' - rule given by default. All words are written in small letters. And this includes the names of the presidents. 

# ### The most important objects of the Tokenizer class are 'word_index' and 'word_counts'.

# In[47]:


print ("trump word total ", tokenizer.word_counts['trump'], " times appears in the text")
print ("trump word id = ", tokenizer.word_index['trump'])


# In[48]:


#Let's print out all the words and frequencies
for word in tokenizer.word_index:
    print (word, "=", tokenizer.word_counts[word], "    ")


# ### So far we have tried to represent the input data as numerical data, and now it is time to do so.

# In[49]:


sequences = tokenizer.texts_to_sequences(training_X)
 
for line in sequences:
    print (line, "  ")


# ### So the sentences could be shown in numbers. There are also a number of shortcomings. We see that some sentences turn into empty strings like []. In most cases this happens less frequently. Because waiting behind the words in the text is natural for the daily spoken language.
# 
# 

# ### And the another problem is vectors of different length. It would be better if we keep this at a certain limit for most algorithms. For this we will use the tokenizer_padding method.

# In[53]:


# the length of each input data is only 4
from keras.preprocessing.sequence import pad_sequences
 
entry = pad_sequences(sequences, maxlen=4)


# In[54]:


for line in entry:
    print (line, "  ")


# ### And happy ending! Now the input data is ready according to the text preprocessing conditions we set.
# 
# ### The preparation of this data set is an important foundation for machine learning and deep learning models. And after that, other further steps can be taken.

# In[ ]:




