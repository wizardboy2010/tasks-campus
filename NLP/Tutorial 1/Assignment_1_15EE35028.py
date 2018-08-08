import nltk
from nltk.corpus import brown
from nltk import ngrams
from collections import Counter
import re
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np

################################################################################# Preprocessing

def clean(sent_list):
    sent_list = [i.lower() for i in sent_list]
    sent_pre = [re.sub('[^A-Za-z]', '', i) for i in sent_list]
    return [i for i in sent_pre if i != '']

corpus_preproc = [clean(i) for i in brown.sents()[:40000]]

#################################################################################  Models

def unigram_model(corpus):
    unigrams = [j for i in corpus for j in i]
    unigrams_count = Counter(unigrams)
    unigrams_count['<s>'] = 40000
    unigrams_count['</s>'] = 40000
    unigrams_count = {i:unigrams_count[i]/len(unigrams) for i in unigrams_count}
    return unigrams_count

def bigram_model(corpus):
    model = {}
    for i in corpus:
        for w1, w2 in ngrams(i, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if w1 not in model:
                model[w1] = []
            model[w1].append(w2)
    model = {i:Counter(model[i]) for i in model}
    model = {i:{j:model[i][j]/sum(model[i].values()) for j in model[i]} for i in model}
    return model

def trigram_model(corpus):
    model = {}
    for i in corpus:
        for w1, w2, w3 in ngrams(i, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if (w1, w2) not in model:
                model[(w1, w2)] = []
            model[(w1, w2)].append(w3)
    model = {i:Counter(model[i]) for i in model}
    model = {i:{j:model[i][j]/sum(model[i].values()) for j in model[i]} for i in model}
    return model

unigram_brown = unigram_model(corpus_preproc)
bigram_brown = bigram_model(corpus_preproc)
trigram_brown = trigram_model(corpus_preproc)

##################################################################################  Ziphs law
#############################
unigrams = [j for i in corpus_preproc for j in i]
unig_count = Counter(unigrams)
unig_count['<s>'] = 40000
unig_count['</s>'] = 40000
unig_count = list(reversed(sorted(unig_count.items(), key = itemgetter(1))))
del unigrams

plt.title("Ziph's law for unigram")
plt.plot([i[1] for i in unig_count[:50]])
plt.show()

#########################################
bigrams = []
for i in corpus_preproc:
    bigrams += [i for i in ngrams(i, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')]
bigrams = [(' ' if j is None else j for j in i) if (None in i) else i for i in bigrams]
bi_count = Counter(bigrams)
bi_count = list(reversed(sorted(bi_count.items(), key = itemgetter(1))))
del bigrams

plt.title("Ziph's law for bigram")
plt.plot([i[1] for i in bi_count[:50]])
plt.show()

###########################################
trigrams = []
for i in corpus_preproc:
    trigrams += [i for i in ngrams(i, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')]
trigrams = [(' ' if j is None else j for j in i) if (None in i) else i for i in trigrams]
tri_count = Counter(trigrams)
tri_count = list(reversed(sorted(tri_count.items(), key = itemgetter(1))))
del trigrams

plt.title("Ziph's law for Trigram")
plt.plot([i[1] for i in tri_count[:50]])
plt.show()

################################################################################# Top 10

print('Top 10 for Unigram')
for i, j in unig_count[:10]:
    print(i, j)

print('Top 10 for Bigram')
for i, j in bi_count[:10]:
    print(i, j)
    
print('Top 10 for Trigram')
for i, j in tri_count[:10]:
    print(i, j)
    
################################################################################# Testing
    
f = open("test_examples.txt",'r')
test = [i. strip() for i in f.readlines()]
test_clean = [clean(i.split()) for i in test]

print('For Unigram')
for sent in test_clean:
    try:
        prob = np.prod([unigram_brown[j] for j in sent])
        print('\nFor sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\t Perplexity:',(1/prob)**(1/len(sent)))
    except:
        print('\nFor sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood: Inf\tPerplexity:Inf')
        
print('For Bigram')
for sent in test_clean:
    try:
        prob = np.prod([bigram_brown[i][j] for i, j in ngrams(sent, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')])
        print('\nFor sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\tPerplexity:',(1/prob)**(1/len(sent)))
    except:
        print('\nFor sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood: Inf\tPerplexity:Inf')
        
print('For Trigram')
for sent in test_clean:
    try:
        prob = np.prod([trigram_brown[(i, j)][k] for i, j, k in ngrams(sent, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')])
        print('\nFor sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\tPerplexity:',(1/prob)**(1/len(sent)))
    except:
        print('\nFor sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood: Inf\tPerplexity:Inf')

del unigram_brown, bigram_brown, trigram_brown
###################################################################################
#### Task 2
###################################################################################
        
def unigram_model_adsm(corpus, k):
    unigrams = [j for i in corpus for j in i]
    unigrams_count = Counter(unigrams)
    unigrams_count['<s>'] = 40000
    unigrams_count['</s>'] = 40000
    N = len(unigrams_count)
    unigrams_count = {i:(k+unigrams_count[i])/(k*N+len(unigrams)) for i in unigrams_count}
    return unigrams_count, N

def bigram_model_adsm(corpus,k):
    model = {}
    for i in corpus:
        for w1, w2 in ngrams(i, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if w1 not in model:
                model[w1] = []
            model[w1].append(w2)
    model = {i:Counter(model[i]) for i in model}
    N = sum([len(model[i]) for i in model])
    model = {i:{j:(k+model[i][j])/(k*N+sum(model[i].values())) for j in model[i]} for i in model}
    return model, N

def trigram_model_adsm(corpus,k):
    model = {}
    for i in corpus:
        for w1, w2, w3 in ngrams(i, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if (w1, w2) not in model:
                model[(w1, w2)] = []
            model[(w1, w2)].append(w3)
    model = {i:Counter(model[i]) for i in model}
    N = sum([len(model[i]) for i in model])
    model = {i:{j:(k+model[i][j])/(k*N+sum(model[i].values())) for j in model[i]} for i in model}
    return model, N


print('\nfor Unigram')
for k in [0.0001, 0.001, 0.01, 0.1, 1]:
    print('\nFor k=', k)
    unigram_adsm, N = unigram_model_adsm(corpus_preproc, k)
    for sent in test_clean:
        prob = []
        for j in sent:
            try:
                prob.append(unigram_adsm[j])
            except:
                prob.append(1/N)
        prob = np.prod(prob)
        print('For sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\t Perplexity:',(1/prob)**(1/len(sent)))

print('\nfor Bigram')
for k in [0.0001, 0.001, 0.01, 0.1, 1]:
    print('\nFor k=', k)
    bigram_adsm, N = bigram_model_adsm(corpus_preproc, k)
    for sent in test_clean:
        prob = []
        for i, j in ngrams(sent, 2, pad_left=True, pad_right=True):
            try:
                prob.append(bigram_adsm[i][j])
            except:
                prob.append(1/N)
        prob = np.prod(prob)
        print('For sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\tPerplexity:',(1/prob)**(1/len(sent)))

        
print('\nfor Trigram')
for k in [0.0001, 0.001, 0.01, 0.1, 1]:
    print('\nFor k=', k)
    trigram_adsm, N = trigram_model_adsm(corpus_preproc, k)
    for sent in test_clean:
        prob = []
        for i, j, k in ngrams(sent, 3, pad_left=True, pad_right=True):
            try:
                prob.append(trigram_adsm[(i, j)][k])
            except:
                prob.append(1/N)
        prob = np.prod(prob)
        print('For sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\tPerplexity:',(1/prob)**(1/len(sent)))

del unigram_adsm, bigram_adsm, trigram_adsm

##################################################################################
####  Task 4
##################################################################################

def unigram_model(corpus):
    unigrams = [j for i in corpus for j in i]
    unigrams_count = Counter(unigrams)
    unigrams_count = {i:unigrams_count[i]/len(unigrams) for i in unigrams_count}
    return unigrams_count

def bigram_model(corpus):
    model = {}
    for i in corpus:
        for w1, w2 in ngrams(i, 2, pad_left=True, pad_right=True):
            if w1 not in model:
                model[w1] = []
            model[w1].append(w2)
    model = {i:Counter(model[i]) for i in model}
    model = {i:{j:model[i][j]/sum(model[i].values()) for j in model[i]} for i in model}
    return model

unigram_brown = unigram_model(corpus_preproc)
bigram_brown = bigram_model(corpus_preproc)

def interpolation(i, j, lam):
    try:
        p_bi = bigram_brown[i][j]
    except:
        p_bi = 0
    try:
        p_uni = unigram_brown[i]
    except:
        p_uni = 0
    return lam*p_bi+(1-lam)*p_uni

for lam in [0.2, 0.5, 0.8]:
    print('\nFor lamda =', lam,'\n')
    for sent in test_clean:
        prob = np.prod([interpolation(i, j, lam) for i, j in ngrams(sent, 2, pad_left=True, pad_right=True)])
        if prob == 0:
            print('For sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood: Inf\tPerplexity:Inf')
        else:
            print('For sentence..', ' '.join(sent) ,'\nprobability:', prob,' \tlog-likelihood:', np.log10(prob),'\tPerplexity:',(1/prob)**(1/len(sent)))

