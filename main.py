
from laws_data import laws

import re
import json
import pandas as pd
import numpy as np

# sparse retrieval
import bm25s

# tokenize
import jieba

# dense retrieval
from transformers import AutoTokenizer, AutoModel

# query expansion
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')


def my_split(s):
    # remove repeated words
    s = " ".join(dict.fromkeys(s.split()))
    return ' '.join(jieba.cut_for_search(s))

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()  # 使用 [CLS] token 的嵌入向量
def cosine_similarity(vec1, vec2):
    return (np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))[0][0]

def query_expan(query):
    global laws    

    output_query = " ".join(dict.fromkeys(query.split()))

    synonyms = set()
    for word in output_query.split(' '):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    
    output_query += ' '.join(synonyms)
    return output_query

def unit_search(query):
    global retriever, tokenizer, model
    answer = ""

    # Query expansion
    query = query_expan(query)

    # Sparse retrieval
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=200)
    results = results[0]
    scores = scores[0]
    max_score = max(scores)

    # Dense retrieval
    query_embedding = encode_text(query, tokenizer, model)
    candidate_embeddings = [encode_text(result, tokenizer, model) for result in results]
    
    similarity_scores = []
    threshold = np.average(scores)
    for doc_embedding, sparse_score in zip(candidate_embeddings, scores):
        if sparse_score <= threshold: 
            similarity_scores.append(0)
        else:
            sparse_score /= max_score
            cos_score = cosine_similarity(query_embedding, doc_embedding)
            similarity_scores.append(sparse_score + cos_score)


    reordered_indices = np.argsort(similarity_scores)[::-1]

    for rank, idx in enumerate(reordered_indices):
        if rank >= 20: break
        if similarity_scores[idx] < 1: continue
        doc = results[idx]
        match = re.match(r"(.*?)(\+-\+-)", doc).group(1)
        match = ''.join(match.split(' '))
        if len(answer): answer +=  "," + match
        else: answer = match

    return answer

def run():
    global laws, retriever, tokenizer, model
    laws = [my_split(doc) for doc in laws]

    # initialize sparse retriever
    tokenized_laws = bm25s.tokenize(laws)
    retriever = bm25s.BM25(corpus=laws)
    retriever.index(tokenized_laws)

    # initialize dense retriever (LegalBERT)
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    # query = "【時事】假房東與無權處分法律問題兩名男子在去年12月在臉書上找到一間位於台北市大安區的公寓，以每月二萬元的租金與洪姓女子簽訂租約，並一次付清八個月的房租和四萬元的押金，共計二十萬元。今年七月，兩名男子發現洪姓女子已經失聯，而且有一對自稱是房屋代管的男女來到門口，指控他們侵入住居，並要求他們立即搬離。兩名男子向警方報案，警方調查發現，洪姓女子是假房東，她使用偽造的權狀來欺騙房客，並收取不當利益。警方已掌握洪姓女子的身分資料，正傳喚她到案說明，全案朝詐欺、偽造文書等罪嫌偵辦中。請問有關民法上無權處分與善意取得的法律效果：在假房東案件中，真房東、假房東和房客各自的權利和義務是什麼？他們之間的法律關係如何判斷？如果房客在租屋時知道或應該知道假房東沒有出租的權利，他們還能否主張善意取得房屋使用權？為什麼？如果真房東在假房東出租房屋後，承認了假房東對房客的處分，那麼房客是否能繼續居住房屋？為什麼？"
    # 中華民國刑法第210條,民法第170條,民法第179條,民法第184條,民法第213條,民法第226條,民法第421條,民法第423條,民法第767條

    # Query the corpus and get top-k results
    with open("./test_data.jsonl", "r", encoding="utf-8") as f:
        answers = {}

        for line in f:
            entry = json.loads(line.strip())
            
            title = entry["title"]
            question = entry["question"]
            if question is None: question = ""

            query = title + question
            query = my_split(query)

            answers[entry['id']] = unit_search(query)
            print("finish", entry['id'])
        df = pd.DataFrame(answers.items(), columns=["id", "TARGET"])
        df.to_csv(f'submission_sparse_dense.csv', index=False)

if __name__ == "__main__":
    run()