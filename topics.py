"""BBC topic modeling with correct token filtering and best-model retention."""
import argparse,json,math,re
from pathlib import Path
from itertools import product
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

stemmer=PorterStemmer()
def preprocess(text):
    words=re.findall(r'[a-z]+',str(text).lower())
    return [stemmer.stem(word) for word in words if len(word)>2 and word not in ENGLISH_STOP_WORDS]

def train(texts):
    documents=[preprocess(text) for text in texts]
    dictionary=Dictionary(documents);dictionary.filter_extremes(no_below=5,no_above=.7)
    if len(dictionary)<10:raise ValueError('Need a larger corpus with at least ten usable terms.')
    corpus=[dictionary.doc2bow(doc) for doc in documents]
    candidates=[];best_model=None;best_score=-math.inf
    for count,alpha,eta in product([3,5,7],['symmetric','asymmetric'],['symmetric','auto']):
        model=LdaModel(corpus=corpus,id2word=dictionary,num_topics=count,alpha=alpha,eta=eta,
                       random_state=42,passes=10,chunksize=100,iterations=50)
        score=float(CoherenceModel(model=model,texts=documents,dictionary=dictionary,coherence='c_v',topn=10,processes=1).get_coherence())
        if not math.isfinite(score):raise RuntimeError('Non-finite topic coherence.')
        candidates.append({'topics':count,'alpha':alpha,'eta':eta,'coherence':score})
        if score>best_score:best_score=score;best_model=model
    best=max(candidates,key=lambda c:c['coherence'])
    assert best_model.num_topics==best['topics']
    lda_topics=[[word for word,_ in best_model.show_topic(i,topn=10)] for i in range(best_model.num_topics)]
    vectorizer=TfidfVectorizer(tokenizer=preprocess,token_pattern=None,lowercase=False,max_df=.7,min_df=5)
    matrix=vectorizer.fit_transform(list(texts))
    nmf=NMF(n_components=best['topics'],init='nndsvda',random_state=42,max_iter=500)
    nmf.fit(matrix);vocab=vectorizer.get_feature_names_out()
    nmf_topics=[[str(vocab[i]) for i in component.argsort()[-10:][::-1]] for component in nmf.components_]
    return {'dataset':'BBC supplied news CSV','documents':len(documents),'seed':42,'selection_metric':'training-corpus c_v coherence',
            'grid':candidates,'best_parameters':best,'displayed_model_is_best':True,'lda_topics':lda_topics,'nmf_topics':nmf_topics,
            'nmf_reconstruction_error':float(nmf.reconstruction_err_),'limitation':'Topic coherence and reconstruction error are different metrics; neither proves human-readable or factual topics.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--text-column',default='description');p.add_argument('--output',type=Path,default=Path('metrics.json'))
    a=p.parse_args();df=pd.read_csv(a.data)
    if a.text_column not in df:p.error('Text column not found: '+a.text_column)
    texts=df[a.text_column].dropna().drop_duplicates().tolist()
    r=train(texts);a.output.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(r['best_parameters']))
if __name__=='__main__':main()
