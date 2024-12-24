# IRIE2024_group3 Traditional Approach
The term project of NTU IRIE2024 Group3.
This is the repo for traditional approach. The LLM-based please refer to https://github.com/Eugene-Liao/IRIE2024_group3.

## Introduction
### Data pre-processing
For the data pre-processing of traditional approach, we directly concatenate each article name (law name) with its associated documents, including the law content and provided train data.
The process begins by unzipping the law files and extracting the law names (e.g., ”XX法第XX條之X”) along with their corresponding content. These are stored in a dictionary, where the keys represent the law names, and the values are their content.
Next, we process the training data provided by TA. For each entry, the label is used to identify the relevant provisions. The title and question of the entry are concatenated and appended to the content of the corresponding provisions.
After preparing all the necessary data, the Chinese word segmentation tool, Jieba, is used to tokenize the content. Finally, the law names are concatenated with their tokenized content to form the input for subsequent sparse retrieval tasks.
We also experimented with another data pre-processing approach, which involved concatenating the law content and training data separately with the law names. However, we did not find an effective normalization method for the retrieved results from this approach.

### Retrieval
We experimented with a series of traditional approaches, including query expansion (WordNet), sparse retrieval (BM25) and dense retrieval (LegalBERT).
We first applied the Chinese word segmentation tool, Jieba, to tokenize the query for improved processing of Chinese text. Subsequently, WordNet is employed to identify synonyms of the query terms, enabling query expansion. An initial retrieval was then performed using BM25, limiting the result set to 200 documents. Documents with relevance score lower than the average were filtered out. Finally, we used LegalBERT to compute the embeddings for the query and the documents, evaluating their relevance using consine similarity. The the relevance score were then used to rerank the initial retrieved document set. Extract the law, with the
top 20 documents output as the final result.

## Requirement
```bash
ftfy: 6.3.1
striprtf: 0.0.28
pandas: 2.2.3
numpy: 1.26.4
bm25s: 0.2.5
jieba: 0.42.1
transformers: 4.47.1
nltk: 3.9.1
```

## Usage
- Data pre-processing:
    ```bash
    python extract_law.py
    ```
- Retrieval"
    ```bash
    python main.py
    ```
