from gensim import corpora, models, similarities
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from collections import defaultdict
import nltk
import json
import re
import pprint

# open json file
json_data = open('documents.json').read()
data = json.loads(json_data)
origin_documents = defaultdict(list)
documents = defaultdict(list)

# store paragraph in dictionary
for document in data:
    for paragraph in document['text']:
        origin_documents[document['docid']].append(paragraph)
        paragraph = re.sub(r'[^\w%\$-]', ' ', paragraph)
        documents[document['docid']].append(paragraph)
        
para_doc = {}
doc_para = defaultdict(list)
# store paragraph in a list
para_list = []
for docid in documents.keys():
    for paragraph in documents[docid]:
        para_doc[len(para_list)] = docid
        para_list.append(paragraph)
origin_para_list = []
for docid in origin_documents.keys():
    for paragraph in origin_documents[docid]:
        origin_para_list.append(paragraph)

for paraid in para_doc.keys():
    doc_para[para_doc[paraid]].append(paraid)
    
# delete stop words
stoplist = nltk.corpus.stopwords.words('english')
raw_texts = []
raw_texts = [[word for word in para.lower().split() if word not in stoplist] for para in para_list]

# lemmatize the word in document
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

def lemma_word(word):
    lemmas = lemmatizer(word, u'VERB')
    if lemmas[0] == word:
        lemmas = lemmatizer(word, u'NOUN')
        if lemmas[0] == word:
            lemmas = lemmatizer(word, u'ADJ')
    return lemmas[0]

lemma_texts = []
lemma_texts = [[lemma_word(word) for word in text] for text in raw_texts]

# bulid phrase model
phrases = Phrases(lemma_texts)
bigram = Phraser(phrases)

# connect phrase word by the phrase model
texts = []
for text in raw_texts:
    texts.append(bigram[text])
    
# store words as paragraph
para_texts = []
for docid in doc_para.keys():
    paralist = doc_para[docid]
    para_text = []
    for paraid in paralist:
        para_text.append(texts[paraid])
    para_texts.append(para_text)
    
dictionarys = []
lsi_models = []
corpus_lsis = []
for para_text in para_texts:
    # build tfidf model
    dictionary = corpora.Dictionary(para_text)
    corpus = [dictionary.doc2bow(text) for text in para_text]
    tfidf = models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]
    # build lsi model
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
    corpus_lsi = lsi[corpus]
    dictionarys.append(dictionary)
    lsi_models.append(lsi)
    corpus_lsis.append(corpus_lsi)
    
  from scipy import spatial
import re
# calculate cosine similarity between word1 and word2
# vector must be the same size
def get_cosine_similarity(word1, word2):
    vec1 = []
    for word in word1:
        vec1.append(word[1])
    vec2 = []
    for word in word2:
        vec2.append(word[1])
    if vec1 == []:
        print (word1)
    return 1 - spatial.distance.cosine(vec1, vec2)

# get the lsi vector for text
def get_vector(text, docid):
    text = re.sub(r'[^\w%\$-]', ' ', text)
    text = [word for word in text.lower().split() if word not in stoplist]
    text = bigram[text]
    vec_bow = dictionarys[docid].doc2bow(text)
    vec_lsi = lsi_models[docid][vec_bow]
    return vec_lsi

# get the lsi vector for text
def get_question_vector(text, docid, stopwords):
    text = re.sub(r'[^\w%\$-]', ' ', text)
    text = [word for word in text.lower().split() if word not in stopwords]
    text = bigram[text]
    vec_bow = dictionarys[docid].doc2bow(text)
    vec_lsi = lsi_models[docid][vec_bow]
    return vec_lsi

# input: a lsi vector of question 
# output: a list wich contains tuple (paraid, similarity) with the top 5 similarity
def find_para(question_vec, docid):
    top_size = 3
    max_result = []
    for paraid in doc_para[docid]:
        para_vec_id = doc_para[docid].index(paraid)
        para_vec = corpus_lsis[docid][para_vec_id]
        similarity = get_cosine_similarity(para_vec, question_vec)
        #max_cosine_similarity.append(similarity)
        #max_docid.append(docid)
        max_result.append((paraid, similarity))
        if len(max_result) > top_size:
            max_result.sort(key=lambda tup: tup[1])
            max_result.pop(0)
    para_result = []
    for paraid, value in max_result:
        para_result.append(paraid)
        #para_result.append(doc_para[docid].index(paraid))
    return para_result
    
 # split paragraph into sentence
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_sent(question_vec, docid, paraids):
    top_size = 5
    max_result = []
    for paraid in paraids:
        sentlist = tokenizer.tokenize(origin_para_list[paraid])
        for sent in sentlist:
            sent_vec = get_vector(sent, docid)
            if sent_vec != []:
                similarity = get_cosine_similarity(sent_vec, question_vec)
                max_result.append((sent, similarity))
                if len(max_result) > top_size:
                    max_result.sort(key=lambda tup: tup[1])
                    max_result.pop(0)
    sent_result = []
    for sent, value in max_result:
        sent_result.append(sent)
    return sent_result
    
# store qustion and docid in list
json_data = open('devel.json').read()
training = json.loads(json_data)
questions = []
docids = []
#question_ids = []
answers = []

for item in training:
    questions.append(item['question'])
    docids.append(item['docid'])
    #question_ids.append(item['id'])
    answers.append(item['text'])
    
# get the related sentence for the answer
stopwords = nltk.corpus.stopwords.words('english')
additional_stopwords = ['much', 'many', 'large', 'far', 'long', 'time', 'date', 'value', 'country', 'nation']
stopwords.append(additional_stopwords)

def get_sent_result(question_id):
    question = questions[question_id]
    docid = docids[question_id]
    question_vec = get_question_vector(question, docid, stopwords)
    sent_result = []
    
    if len(question_vec) != 0:
        para_result = find_para(question_vec, docid)
        sent_result = get_sent(question_vec, docid, para_result)
    else:
        sent_resut = []
    return sent_result
    
 # get question types
import spacy
import re

cardinal = {'many', 'large', 'far', 'long', 'old', 'big', 'often', 'high'}
money = {'money', 'cost', 'budget', 'spend', 'spent', 'investment', 'invest', 'invested',
         'revenue', 'wage', 'price', 'worth', 'pay', 'paid', 'spend', 'spent', 'income'}

def get_question_type(question):
    question_lower = question.lower()
    question_lower = re.sub(r'[^\w]', ' ', question_lower)
    question_lower = question_lower.lstrip().rstrip()
    type = 'OTHERS'
    if bool(re.search(r'^when\b', question_lower) or re.search(r'^since when\b', question_lower) 
            or re.search(r'\bwhen$', question_lower)):
        type = 'DATE'
    elif bool(re.search(r'^why\b', question_lower) or re.search(r'\bwhy$', question_lower)):
        type = 'REASON'
    elif bool(re.search(r'^who\b', question_lower) or re.search(r'^whom\b', question_lower)
             or re.search(r'\bwho$', question_lower) or re.search(r'\bwhom$', question_lower)):
        type = 'PERSON'
    elif bool(re.search(r'^where\b', question_lower) or re.search(r'\bwhere$', question_lower)):
        type = 'GPE'
    elif bool(re.search(r'\bwhat\b', question_lower)):
        if bool("what country" in question_lower or "what nation" in question_lower):
            type = 'GPE'
        elif bool("what century" in question_lower):
            type = 'DATE'
        elif bool("what period" in question_lower):
            type = 'DATE'
        elif bool("what time" in question_lower or "what years" in question_lower):
            type = 'DATE'
        elif bool('percent' in question_lower or 'rate' in question_lower):
            type = 'PERCENT'
        elif bool(re.search(r'\bvalue\b', question_lower)):
            type = 'CARDINAL'
        elif bool("the amount of" in question_lower or "the value of" in question_lower):
            type = 'CARDINAL'
            for word in question_lower.split():
                if word in money:
                    type = 'MONEY'
        elif bool(re.search(r'\b(year|month|day|date|decade)', question_lower)):
            type = 'DATE'
        elif bool(re.search(r'\bpopulation\b', question_lower) or re.search(r'\bnumber\b', question_lower)
                 or bool(re.search(r'\baverage\b', question_lower)) or bool(re.search(r'\bage\b', question_lower))):
            type = 'CARDINAL'
        else:
            type = "NN"
    elif bool(re.search(r'\bhow\b', question_lower)):
        if bool("how much" in question_lower):
            if bool('how much of' in question_lower):
                type = 'PERCENT'
            else:
                type = 'CARDINAL'
        elif bool("how many" in question_lower):
            if bool('how many of' in question_lower):
                type = 'PERCENT'
            else:
                type = 'CARDINAL'
        elif bool("how old" in question_lower):
            type = 'CARDINAL'
        elif bool("how long" in question_lower):
            type = 'HOWLONG'
        else:
            for word in question_lower.split():
                if word in cardinal:
                    type = 'CARDINAL'
    elif bool(re.search(r'\bwhich\b', question_lower)):
        if bool("which country" in question_lower or "which nation" in question_lower):
            type = 'GPE'
        elif bool("which century" in question_lower):
            type = 'DATE'
        elif bool("which period" in question_lower):
            type = 'DATE'
        elif bool(re.search(r'\b(year|month|day|date|decade)\b', question_lower)):
            type = 'DATE'
        else:
            type = "NN"
    if type == 'OTHERS':
        if bool(re.search(r'\b(rank|ranking)\b', question_lower)):
            type = 'ORDINAL'
        elif bool(re.search(r'\bwho\b', question_lower) or re.search(r'\bwhom\b', question_lower)
                 or re.search(r'\bwhose\b', question_lower)):
            type = 'PERSON'
        elif bool(re.search(r'\bwhen\b', question_lower)):
            type = 'DATE'
        elif bool(re.search(r'\bwhere\b', question_lower)):
            type = 'GPE'
        elif bool(re.search(r'\bwhy\b', question_lower)):
            type = 'REASON'
        elif bool(re.search(r'\bhow\b', question_lower)):
            type = 'METHOD'
        elif bool(re.search(r'\bname\b', question_lower)):
            type = 'NN'
    return type
    
import sys
!{sys.executable} -m spacy download en_core_web_sm

# find answer based on question type
from spacy.pipeline import EntityRecognizer
from spacy.pipeline import Tagger
import calendar

nlp = spacy.load('en_core_web_sm')
ner = EntityRecognizer(nlp.vocab)
#ner.from_disk('D:/Software/Python/Lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')
ner = spacy.load('en_core_web_sm')

tagger = Tagger(nlp.vocab)
#tagger.from_disk('D:/Software/Python/Lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.1')
tagger= spacy.load('en_core_web_sm')

def sort_answer(answer):
    sort_list = []
    for token in answer:
        sort_list.append(token.text)
    return sorted(set(sort_list), key=sort_list.index)
    

def get_date_answer(sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'DATE':
                #print token
                if bool(re.search(r'[0-9+]',token.text)) or bool(token.text in calendar.month_name):
                    answer.append(token)
    return sort_answer(answer)

def get_howlong_answer(sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'DATE':
                answer.append(token)
    if len(answer) == 0:
        for sentence in sentences:
            doc = nlp(sentence)
            processed = ner(doc)
            for token in processed:
                if token.ent_type_ == 'QUANTITY':
                    answer.append(token)
    if len(answer) == 0:
        for sentence in sentences:
            doc = nlp(sentence)
            processed = ner(doc)
            for token in processed:
                if token.ent_type_ == 'CARDINAL':
                    answer.append(token)
    return sort_answer(answer)

def get_person_answer(question, sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'PERSON' or token.ent_type_ == 'ORG':
                #print token
                answer.append(token)
    if len(answer) == 0:
        return get_nn_answer(question, sentences)
    else:
        return sort_answer(answer)

def get_gpe_answer(sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'GPE' or token.ent_type_ == 'NORP':
                #print token
                answer.append(token)
    return sort_answer(answer)

def get_percent_answer(sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'PERCENT':
                #print token
                answer.append(token)
    if len(answer)==0:
        for sentence in sentences:
            doc = nlp(sentence)
            processed = ner(doc)
            for token in processed:
                if token.ent_type_ == 'CARDINAL':
                    #print token
                    answer.append(token)
    return sort_answer(answer)

def get_cardinal_answer(sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'CARDINAL' or token.ent_type_ == 'QUANTITY':
                #print token
                answer.append(token)
    if len(answer)==0:
        for sentence in sentences:
            doc = nlp(sentence)
            processed = ner(doc)
            for token in processed:
                if token.ent_type_ == 'MONEY' or token.ent_type_ == 'PERCENT':
                    #print token
                    answer.append(token)
    return sort_answer(answer)

def get_money_answer(sentences):
    answer = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed = ner(doc)
        for token in processed:
            if token.ent_type_ == 'MONEY':
                #print token
                answer.append(token)
    if len(answer)==0:
        for sentence in sentences:
            doc = nlp(sentence)
            processed = ner(doc)
            for token in processed:
                if token.ent_type_ == 'CARDINAL' or token.ent_type_ == 'QUANTITY':
                    #print token
                    answer.append(token)
    return sort_answer(answer)

reason_pp = ['because', 'since', 'because of', 'in order to', 'due to', 'for',
             'to', 'cause', 'in view of', 'owing to']
reason_reg = [r"\bbecause\b", r"\bsince\b", r"\bbecause of\b", r"\bin order to\b",
              r"\bdue to\b", r"\bfor\b", r"\bto\b", r"\bcause\b", r"\bin view of\b", r"\bowing to\b"]

def get_reason_answer(sentences):
    answer = []
    for sentence in sentences:
        pplen = len(reason_pp)
        answer_string = ''
        for ppid in range(pplen):
            if bool(re.search(reason_reg[ppid], sentence)):
                start_pos = re.search(reason_reg[ppid], sentence).start()
                start_pos = start_pos + len(reason_pp[ppid])
                for pos in range(start_pos, len(sentence)):
                    if sentence[pos].isalnum() or sentence[pos]==' ':
                        answer_string = answer_string + sentence[pos]
                    else:
                        break
                break
        answer = answer + answer_string.split()[:5]
    return sorted(set(answer), key=answer.index)

def get_method_answer(sentences):
    answer = []
    return sort_answer(answer)

def get_root(sentence):
    doc = nlp(sentence)
    root = ''
    for token in doc:
        if token.dep_ == 'ROOT':
            root = lemma_word(token.text)
    return root

def get_dep_chunk(sentence, dep):
    doc = nlp(sentence)
    processed = tagger(doc)
    chunklist = []
    for chunk in processed.noun_chunks:
        chunklist.append(chunk.text)
    dep_chunk = ''
    for token in doc:
        if token.dep_ == dep:
            for chunk in chunklist:
                if token.text in chunk:
                    dep_chunk = chunk
            if len(dep_chunk) == 0:
                dep_chunk = token.text
    return dep_chunk

def is_person_date(chunk_word):
    word = nlp(chunk_word)
    processed = ner(word)
    for token in word:
        token_type = token.ent_type_
        if token_type == 'PERSON' or token_type == 'DATE' or token_type == 'TIME':
            return True
    return False

def is_subset(small_list, big_list):
    for element in small_list:
        if element not in big_list:
            return False
    return True

def lemma_string(sentence):
    lemma_sent = re.sub(r'[^\w%\$-]', ' ', sentence)
    lemma_sent = lemma_sent.lower().split()
    lemma_sent = [lemma_word(word) for word in lemma_sent if word not in stoplist]
    return lemma_sent

punctuation = [',','.','!','?',';']
def get_nn_answer(question, sentences):
    question_root = get_root(question)
    lemmatized_root = lemma_string(question_root)
    #print ("question_root:", question_root)
    answer = []
    if question_root != 'be' and question_root != 'have' and len(lemmatized_root) != 0:
        question_keyword = question_root
        question_keyword = lemma_string(question_keyword)
        #print ("question keyword:", question_keyword)
        question_word = lemma_string(question)
        #print (question_word)
        sentences.reverse()
        for i in range(len(sentences)):
            sentence = sentences[i]
            lemma_sent = lemma_string(sentence) 
            #print ("sentence:",lemma_sent)
            if is_subset(question_keyword, lemma_sent):
                doc = nlp(sentence)
                processed = tagger(doc)
                chunklist = []
                for chunk in processed.noun_chunks:
                    if not is_person_date(chunk.text):
                        chunklist.append(chunk.text)
                start = False
                for token in doc:
                    word = token.text
                    lemmitized_word = lemma_word(word)
                    if start:
                        if word not in punctuation:
                            for chunk in chunklist:
                                if word in chunklist:
                                    #print ("chunk:",chunk)
                                    answer.append(chunk)
                        else:
                            start = False
                    elif lemmitized_word == question_keyword[0]:
                        start = True
            if i == 2 and len(answer) != 0:
                break
    if question_root == 'be' or question_root == 'have' or len(lemmatized_root) == 0 or len(answer) == 0:
        #print ("answer = 0")
        question_keyword = get_dep_chunk(question, 'nsubj')
        if len(question_keyword) == 0:
            question_keyword = get_dep_chunk(question, 'dobj')
            if len(question_keyword) == 0:
                question_keyword = get_dep_chunk(question, 'pobj')
        question_keyword = lemma_string(question_keyword)
        #print ("question keyword:",question_keyword)
        
        answer_size = 10
        question_word = lemma_string(question)
        #print (question_word)
        sentences.reverse()
        for i in range(len(sentences)):
            sentence = sentences[i]
            lemma_sent = lemma_string(sentence) 
            #print ("sentence:",lemma_sent)
            if is_subset(question_keyword, lemma_sent):
                doc = nlp(sentence)
                processed = tagger(doc)
                for chunk in processed.noun_chunks:
                    if not is_person_date(chunk.text):
                        chunk_word = chunk.text.lower().split()
                        chunk_word = [lemma_word(word) for word in chunk_word]
                        question_set = set(question_word)
                        chunk_set = set(chunk_word)
                        #print ("chunk set:",chunk_set)
                        if len(chunk_set.intersection(question_set)) == 0:
                            for token in chunk:
                                answer.append(token.text)
                            if len(answer) > answer_size:
                                return sorted(set(answer), key=answer.index)
            if i == 2 and len(answer) != 0:
                break
        if len(answer) == 0:
            for sentence in sentences:
                doc = nlp(sentence)
                processed = tagger(doc)
                for chunk in processed.noun_chunks:
                    if not is_person_date(chunk.text):
                        chunk_word = chunk.text.lower().split()
                        chunk_word = [lemma_word(word) for word in chunk_word]
                        question_set = set(question_word)
                        chunk_set = set(chunk_word)
                        #print ("chunk set:",chunk_set)
                        if len(chunk_set.intersection(question_set)) == 0:
                            for token in chunk:
                                answer.append(token.text)
                            if len(answer) > answer_size:
                                return sorted(set(answer), key=answer.index)
    return sorted(set(answer), key=answer.index)
    
# put answer in question_answer list
length = len(questions)
question_answer = []
for i in range(length):
    question = questions[i]
    print (question)
    question_type = get_question_type(question)
    #print (question_type)
    sent_result = get_sent_result(i)
    #print (sent_result)
    if question_type == 'DATE':
        answer = get_date_answer(sent_result)
    elif question_type == 'PERSON':
        answer = get_person_answer(question, sent_result)
    elif question_type == 'PERCENT':
        answer = get_percent_answer(sent_result)
    elif question_type == 'GPE':
        answer = get_gpe_answer(sent_result)
    elif question_type == 'CARDINAL':
        answer = get_cardinal_answer(sent_result)
    elif question_type == 'HOWLONG' :
        answer = get_howlong_answer(sent_result)
    elif question_type == 'MONEY':
        answer = get_money_answer(sent_result)
    elif question_type == 'REASON':
        answer = get_reason_answer(sent_result)
    elif question_type == 'METHOD':
        answer = get_method_answer(sent_result)
    elif question_type == 'NN':
        answer = get_nn_answer(question, sent_result)
    answer_string = ' '.join(answer)
    answer_string = re.sub(r'["]', ' ', answer_string)
    answer_string = ' '.join(answer_string.lower().split()[:5])
    question_answer.append(answer_string)
    
# output the answer to a csv file
f= open("test_result1.csv","wb+")
f.write('id,answer\n'.encode('ascii')) 
for i in range(length):
    f.write((str(question_ids[i])+','+'"'+question_answer[i]+'"'+'\n').encode('ascii','backslashreplace'))
f.close()

# calculate precision of the answer
predict_answer = []
for answerlist in question_answer:
    predict_answer.append(answerlist.split())

devel_answer = []
for answerlist in answers:
    devel_answer.append(answerlist.split())
    
def evaluate(doc1, doc2):
    tp = 0
    fp = 0
    fn = 0
    count = 0
    for i in range(len(doc1)):
        #if get_question_type(questions[i]) == ('DATE' or 'REASON' or 'GPE' or 'PERSON' or 'PERCENT' or 'CARDINAL' or 'ORDINAL'):
        #if get_question_type(questions[i]) == ('NN'):
            answer1 = set(doc1[i])
            answer2 = set(doc2[i])
            right = answer1.intersection(answer2)
            tp = tp + len(right)
            fp = fp + len(answer1) - len(right)
            fn = fn + len(answer2) - len(right)
            count = count + 1
    precision = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    fscore = 2 * precision * recall/(precision + recall)
    print (tp, fp, fn, count)
    return precision, recall, fscore

precision, recall, fscore = evaluate(predict_answer, devel_answer)
print (precision, recall, fscore)
