# -*- coding: utf-8 -*-
"""
Python version: 3.7.3
"""

#%%load packages for LDA

import numpy as np 
#9f6abe9f34efad365694a3ae52d244ebfdc22429
import panda as pd
import system as sys



#%% LDA
def lda_pitchdecks(text_list, number_topics,number_words,counts=[0]):
    '''
    output: the LDA of the text_list with the certain data preprocessing. 
            counts[0] is the number of times this function is called as a index of the output file
            file is saving in the C:\dev\doc\etap_platform
    '''
    # Initialise the count vectorizer
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'Dutch',
                                lowercase = False,
>>>>>>> 755c7779b527c2ea3dfdb9dbf5e6065d7e0cdea8
                                token_pattern = r'\b[a-zA-Z]{3,}\b')
    # generate word counts
    dtm_tf = tf_vectorizer.fit_transform(text_list)
  
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, random_state=0)
    lda.fit(dtm_tf)
    
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    visualization.print_topics(lda, tf_vectorizer, number_words)
    dtm_output=pyLDAvis.sklearn.prepare(lda, dtm_tf, tf_vectorizer)

    outpath = r'C:\dev\doc\etap_platform' 
    print('the output of the html file is in the following location',outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    #run_date=date.today()
    counts[0]+=1
    file_path = os.path.join(outpath, 'LDA_{}.html'.format(counts[0]))
    pyLDAvis.save_html(dtm_output,file_path)
<<<<<<< HEAD


def text_to_wordlist(text):
    review=normalization_word2vec(text)
    words = review.lower().split()
    return words
    
def get_text_string(fpath):    
    #read load all .txt from the folder
    txtfpath=os.path.join(fpath,'*.txt')
    allfiles=glob.glob(txtfpath)
    #print(len(allfiles))
    #print(type(allfiles))
    
    #build corpus doc
    doc=[] # create an empty list
    for fname in allfiles: 
        with open(fname) as f:
            ls=f.readlines() 
            ls=list(map(lambda x:x.strip(),ls))#remove '\n' from ls 
            doc.append(ls)
    corpus=[]#corpus is the corpus of the files
    for i in doc:
        corpus.append(','.join(i)) 
    return ' '.join(map(str, corpus))
=======
>>>>>>> parent of 5bf12f8... add text_to_wordlist function at master branch
