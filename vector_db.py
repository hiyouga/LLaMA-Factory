"""
Reference Docs
[1] https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe
[2] https://js.langchain.com/docs/modules/data_connection/document_transformers/
[3] https://python.langchain.com/docs/integrations/vectorstores/faiss

"""
# %%
import pandas as pd
import os, json
import os.path as osp

from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


### [Step 1] Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
def load_solvook_data(args):
    ## make loader
    db = pd.read_csv(args.db_path)

    mt_loader = DataFrameLoader(db, page_content_column="본문")
    ques_loader = DataFrameLoader(db, page_content_column="질문")
    parap_loader = DataFrameLoader(db, page_content_column="paragraphs")

    mt_data_ = mt_loader.load()
    ques_data_ = ques_loader.load()
    parap_data_ = parap_loader.load()

    data_dict = dict()
    # remove NaN data
    mt_data = [mt_dat for mt_dat in mt_data_ if mt_dat.page_content != "nan"]
    data_dict["mt_data"] = mt_data
    
    ques_data = [ques_dat for ques_dat in ques_data_ if ques_dat.page_content != "nan"]
    data_dict["ques_data"] = ques_data
    
    parap_data = [parap_dat for parap_dat in parap_data_ if parap_dat.page_content != "nan"]
    data_dict["parap_data"] = parap_data

    return data_dict


### [Step 2] Text Split in Loader
## textsplit
def text_split_solvook(args, data_dict):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, add_start_index=True
    )
    docs_dict = dict()
    docs_dict['mt_docs'] = text_splitter.split_documents(data_dict['mt_data'])
    docs_dict['ques_docs'] = text_splitter.split_documents(data_dict['ques_data'])
    docs_dict['parap_docs'] = text_splitter.split_documents(data_dict['parap_data'])

    return docs_dict



### [Step 3] Make 본문, 질문, paragraphs loader to embedding
def make_embedding(args, docs_dict):
    embeddings = OpenAIEmbeddings(openai_api_key=args.openai_api_key, model=args.embedding_model, tiktoken_model_name='cl100k_base')

    ## make embedding
    db_dict = dict()
    if args.search_type in ['similarity', 'similarity_score_threshold', 'mmr']:
        db_dict['mt_db'] = FAISS.from_documents(docs_dict['mt_docs'], embeddings)
        db_dict['ques_db'] = FAISS.from_documents(docs_dict['ques_docs'], embeddings)
        db_dict['parap_db'] = FAISS.from_documents(docs_dict['parap_docs'], embeddings)
    elif args.search_type in ['bm25', 'tf_idf']:
        db_dict['mt_db'] = docs_dict['mt_docs']
        db_dict['ques_db'] = docs_dict['ques_docs']
        db_dict['parap_db'] = docs_dict['parap_docs']
    elif args.search_type == 'sim_bm25':
        db_dict['mt_db'] = [FAISS.from_documents(docs_dict['mt_docs'], embeddings), docs_dict['mt_docs']]
        db_dict['ques_db'] = [FAISS.from_documents(docs_dict['ques_docs'], embeddings), docs_dict['ques_docs']]
        db_dict['parap_db'] = [FAISS.from_documents(docs_dict['parap_docs'], embeddings), docs_dict['parap_docs']]

    return embeddings, db_dict



### [Step 4] Get Top-K docs and bring pairs of the docs
## 1) paragraph - 본문*, 2) 본문 - 본문*, 3) 질문 - 질문* top-K//3 search each
def top_k_search(args, db_dict):
    if args.task in [0, 1, 2]:
        EACH_K_1st = args.top_k // 3
    else:
        EACH_K_1st = args.top_k

    retriever_dict = dict()
    if args.search_type in ['similarity', 'similarity_score_threshold', 'mmr']:
        if args.task in [0, 1, 2]:
            retriever_dict['mt_db_retriever'] = db_dict['mt_db'].as_retriever(search_type=args.search_type, search_kwargs={"k": EACH_K_1st})
            retriever_dict['parap_db_retriever'] = db_dict['parap_db'].as_retriever(search_type=args.search_type, search_kwargs={"k": EACH_K_1st})
        retriever_dict['ques_db_retriever'] = db_dict['ques_db'].as_retriever(search_type=args.search_type, search_kwargs={"k": EACH_K_1st})
        
        
    elif args.search_type == 'bm25':
        from langchain_community.retrievers import BM25Retriever
        if args.task in [0, 1, 2]:
            retriever_dict['mt_db_retriever'] = BM25Retriever.from_documents(db_dict['mt_db'], k=EACH_K_1st)
            retriever_dict['parap_db_retriever'] = BM25Retriever.from_documents(db_dict['parap_db'], k = EACH_K_1st)
        retriever_dict['ques_db_retriever'] = BM25Retriever.from_documents(db_dict['ques_db'], k = EACH_K_1st)
        
            
    elif args.search_type == 'tf_idf':
        from langchain_community.retrievers import TFIDFRetriever
        if args.task in [0, 1, 2]:
            retriever_dict['mt_db_retriever'] = TFIDFRetriever.from_documents(db_dict['mt_db'], k=EACH_K_1st)
            retriever_dict['parap_db_retriever'] = TFIDFRetriever.from_documents(db_dict['parap_db'], k = EACH_K_1st)
        retriever_dict['ques_db_retriever'] = TFIDFRetriever.from_documents(db_dict['ques_db'], k = EACH_K_1st)
        
    elif args.search_type == 'sim_bm25':
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        if args.task in [0, 1, 2]:
            mt_db_retriever_sim = db_dict['mt_db'][0].as_retriever(search_type='similarity', search_kwargs={"k": int(EACH_K_1st//2)})
            mt_db_retriever_bm25 = BM25Retriever.from_documents(db_dict['mt_db'][1], k=int(EACH_K_1st//2))
            retriever_dict['mt_db_retriever'] = EnsembleRetriever(retrievers=[mt_db_retriever_sim, mt_db_retriever_bm25], weight=[0.5, 0.5])
            
            parap_db_retriever_sim = db_dict['parap_db'][0].as_retriever(search_type='similarity', search_kwargs={"k": int(EACH_K_1st//2)})
            parap_db_retriever_bm25 = BM25Retriever.from_documents(db_dict['parap_db'][1], k = int(EACH_K_1st//2))
            retriever_dict['parap_db_retriever'] = EnsembleRetriever(retrievers=[parap_db_retriever_sim, parap_db_retriever_bm25], weight=[0.5, 0.5])
        
        ques_db_retriever_sim = db_dict['ques_db'][0].as_retriever(search_type='similarity', search_kwargs={"k": int(EACH_K_1st//2)})
        ques_db_retriever_bm25 = BM25Retriever.from_documents(db_dict['ques_db'][1], k = int(EACH_K_1st//2))
        retriever_dict['ques_db_retriever'] = EnsembleRetriever(retrievers=[ques_db_retriever_sim, ques_db_retriever_bm25], weight=[0.5, 0.5])
                
    return retriever_dict






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--openai_api_key',type=str, default=None, required=True)
    
    parser.add_argument('--query_path', type=str, default="./data/solvook_handout_te.csv",
                    help="path for solvook_handout_te.csv")
    parser.add_argument('--db_path', type=str, default="./data/solvook_handout_tr.csv",
                    help="path for solvook_handout_tr.csv")
    
    parser.add_argument('--chunk_size', type=int, default=8000)
    parser.add_argument('--chunk_overlap', type=int, default=200)
    
    parser.add_argument('--embedding_model', type=str, default="text-embedding-3-small")
    
    parser.add_argument('--task', type=int, choices=[0, 1, 2, 3, 4], default=0,
                        help="0: all, 1: paragraph, 2: relation, 3: skill, 4: method")
    
    parser.add_argument('--top_k', type=int, default=6)
    parser.add_argument('--search_type', type=str, default='mmr') # opt: similarity, mmr, bm25, tf_idf, sim_bm25
    
    
    args = parser.parse_args()
    
    ## set save_path
    args.result_path = osp.join('./data', 'vector_db.json')
    print(f"Set save path on '{args.result_path}'")
    
    
    ### [Step 1] Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
    print("[Step 1] Load Data!!")
    data_dict = load_solvook_data(args)
    
    ### [Step 2] Text Split in Loader
    print("[Step 2] Text Split!!")
    docs_dict = text_split_solvook(args, data_dict)
    
    ### [Step 3] Make 본문, 질문, paragraphs loader to embedding
    print("[Step 3] Make embedding!!")
    embeddings, db_dict = make_embedding(args, docs_dict)
    
    ### [Step 4] Get Top-K docs and bring pairs of the docs
    ## 1) paragraph - 본문*, 2) 본문 - 본문*, 3) 질문 - 질문* top-K search
    print("[Step 4] Set Top-K search class!!")
    retriever_dict = top_k_search(args, db_dict)
    
    
    with open(args.result_path, 'w') as f : 
	    json.dump(retriever_dict, f, indent=4)