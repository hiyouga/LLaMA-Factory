"""
Reference Docs
[1] https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe
[2] https://js.langchain.com/docs/modules/data_connection/document_transformers/
[3] https://docs.pinecone.io/docs/overview
[4] https://python.langchain.com/docs/integrations/vectorstores/faiss
[5] https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
[6] https://cookbook.openai.com/examples/batch_processing
[7] https://platform.openai.com/docs/guides/batch/getting-started
"""
# %%
import pandas as pd
import json
from tqdm import tqdm
import os.path as osp


### [Step 1] Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
def load_solvook_data(args):
    # load vector db
    with open(args.vector_db_path, "r") as f:
        data_dict = json.load(f)

    # load query
    query_db = pd.read_csv(args.query_path)
    for i in range(len(query_db)):
        query_db.loc[i, 'query'] = f'"본문" : "{query_db.loc[i, "본문"]}", "질문" : "{query_db.loc[i, "질문"]}"'
    
    if args.task == 2:
        query_db = query_db[query_db['relation']!=0].reset_index()

    return data_dict, query_db

    
    
## [Step 2] generation
def generation(args, retriever_dict, query_db):
    print("Start making prompts with top-k contents")
    query_list = list()
    for idx in tqdm(range(len(query_db))):
        #### Top-K search -----------------------------------------------------------------------------------------
        top_ques = retriever_dict['ques_db_retriever'].invoke(query_db['질문'][idx])      # 질문 v.s. 질문*
        if args.task in [1,2]:
            top_mt = retriever_dict['mt_db_retriever'].invoke(query_db['본문'][idx])          # 본문 v.s. 본문*
            top_parap = retriever_dict['parap_db_retriever'].invoke(query_db['본문'][idx])    # paragraph v.s. 본문*
        
        top = top_ques
        if args.task in [1,2]:
            top += top_mt + top_parap
        
        ## Get pair
        top_content = list(); top_metadata = list()
        query_ = dict()
        for k in range(len(top)):
            top_content_ = f"["
            if args.task in [1, 2]:
                top_content_ = f"'본문 id': '{top[k].metadata['textbook_id']}_{top[k].metadata['unit_id']}_{top[k].metadata['story_id']}_{top[k].metadata['paragraph_id']}'. "

                try:
                    try:
                        top_content_ += f"'본문': '{top[k].metadata['paragraphs']}'. "            
                    except:
                        top_content_ += f"'본문': '{top[k].page_content}'. "
                except:
                    pass
            
                try:
                    top_content_ += f"'지문': '{top[k].metadata['본문']}'. "
                except:
                    top_content_ += f"'지문': '{top[k].page_content}'. "
                
                try:
                    if top[k].metadata['relation'] != 0:
                        top_content_ += f" '관계': '{top[k].metadata['relation']}.'" 
                except:
                    pass
                
            elif args.task in [3, 4]:
                try:
                    top_content_ += f"'질문': '{top[k].metadata['질문']}'. "
                except:
                    top_content_ += f"'질문': '{top[k].page_content}'. "
                
                top_content_ += f"'skill': '{top[k].metadata['skill']}'. 'method': '{top[k].metadata['method']}.'"
                
            top_content_ += "]"
            
            top_content.append(top_content_)
            
        top_content = '\n'.join(top_content)
        #### ---------------------------------------------------------------------------------------------------
        
        #### Make prompts --------------------------------------------------------------------------------------
        if args.task == 1:
            # paragraph
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>'지문'과 '질문'을 보고 아래 '후보' 중 어떠한 '본문'과 가장 높은 관련성을 보이는지 하나 골라 해당 '본문'의 '본문 id'를 답하시오. (이 때, id는 1_1_1_1와 같은 형태이다)"
            query_['input'] = f"'지문' : {query_db['본문'][idx]}. '질문' : {query_db['질문'][idx]}."
            query_['input'] += f"\n'후보' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += "<Paragraph>본문 id<Paragraph> 형태로 답하시오. (예를 들어, <Paragraph>1_1_1_1<Paragraph>)"
            query_['output'] = f"{str(query_db['textbook_id'][idx])}_{str(query_db['unit_id'][idx])}_{str(query_db['story_id'][idx])}_{str(query_db['paragraph_id'][idx])}"
        elif args.task == 2:
            # relation
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>'지문'과 '질문'을 보고 아래 '후보' 중 어떠한 '본문'과 가장 높은 관련성을 보이며 어떠한 관계를 갖는지 '보기' 중에 하나 고르시오."
            query_['input'] = f"'지문' : {query_db['본문'][idx]}. '질문' : {query_db['질문'][idx]}."
            query_['input'] += f"\n'후보' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += "'보기' : [1: 원문 (본문의 일부를 변형없이 발췌 혹은 본문 전체를 그대로 지문으로 사용), 2: 삭제 (본문에서 특정 단어/문장을 삭제하여 지문으로 사용), 3: 삽입 (본문에 없던 단어/문장을 추가하여 지문으로 사용 ), 4. 복합 (원문, 삭제, 삽입 관계가 복합적으로 적용)]"
            query_['input'] += "\n <Relation>관계(int)<Relation> 형태로 답하고 (예를 들어, <Relation>1<Relation>), 해당 관계를 고른 이유를 <Description>관계정보를 고른 이유<Description> 형태로 자세히 서술하시오."
            query_['output'] = str(query_db['relation'][idx])
        elif args.task == 3:
            # skill : 문제를 풀기 위해 필요한 능력
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>'참고'를 참고하여 '질문'을 보고 이 문제를 풀기 위한 능력을 '보기' 중에 하나 고르시오."
            query_['input'] = f"'질문' : {query_db['질문'][idx]}"
            query_['input'] += f"\n'참고' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += f"'보기' : [101: 어휘 뜻 이해 (어휘의 뜻을 이해한다.), 102: 영영 풀이 (어휘의 뜻을 영어로 이해한다.), 103: 어휘 혼합, 201: 용법 이해 (용법을 이해한다.), 202: 용법일치불일치 판단 (용법이 서로 같은지 다른지 판단한다.), 203: 문법 혼합, 301: 목적 이해 (글의 목적을 이해한다.), 302: 주제 이해 (글의 주제를 이해한다.), 303: 제목 이해 (글의 제목을 이해한다.), 304: 주장 이해 (글의 주장을 이해한다.), 305: 요지 이해 (글의 요지를 이해한다.), 306: 의미 이해 (글의 의미를 이해한다.), 307: 분위기 이해 (글의 분위기를 이해한다.), 308: 심경 이해 (글의 화자의 심경을 이해한다.), 309: 심경 변화 이해 (글의 화자의 심경 변화를 이해한다.), 310: 어조 이해 (글의 어조를 이해한다.), 311: 순서 이해 (글의 내용을 이해한다.), 312: 대상 이해 (지칭하는 대상을 이해한다), 313: 내용이해 혼합, 401: 내용유추 (글의 내용을 유추한다.), 402: 순서유추 (글의 순서를 유추한다.), 403: 어휘유추 (특정 위치의 어휘를 유추한다.), 404: 연결어유추 (특정 위치의 연결어를 유추한다.), 405: 지칭유추 (지칭하는 대상을 유추한다.), 406: 어휘유추 전반 (유추 내용을 복합적으로 묻는 경우), 407: 내용일치불일치 판단 (내용이 서로 같은지 다른지 판단한다.), 408: 요약 (글을 요약한다.), 409: 번역 (글을 한글로 변역한다.), 410: 영작 (글을 영어로 작문한다.), 411: 내용응용 혼합, 501: 영역통합, 601: 기타]"
            query_['input'] += f"\n<Skill>정답<Skill> 형태로 답하라. (예시, <Skill>405<Skill>)"
            query_['output'] = str(query_db['skill'][idx])
        elif args.task == 4:
            # method : 해당 문제의 '질문'이 학습자의 역량을 검증하기 위해 어떤 방식으로 질문하는지를 의미
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>'참고'를 참고하여 '질문'을 보고 해당 문제가 학습자의 역량을 검증하기 위해 어떠한 방식으로 질문하는지 '보기' 중에 하나 고르시오."
            query_['input'] = f"'질문' : {query_db['질문'][idx]}"
            query_['input'] += f"\n'참고' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += f"'보기' : [1: 맞는 것 찾기(단수) (맞는 것을 찾는다.), 2: 맞는 것 찾기(복수) (맞는 것을 모두 찾는다.), 3: 맞는 것 세기(개수) (맞는 것을 찾아서 개수를 센다.), 4: 틀린 것 찾기(단수) (틀린 것을 찾는다.), 5: 틀린 것 찾기(복수) (틀린 것을 모두 찾는다.), 6: 틀린 것 세기(개수) (틀린 것을 찾아서 개수를 센다.), 7: 다른 것 찾기 (다른 것을 찾는다.), 8: 맞는 위치 찾기 (맞는 위치를 찾는다.), 9: 바른 배열 찾기 (맞는 배열을 찾는다.), 10: 바른 조합 찾기 (맞는 조합을 찾는다.), 11: 어휘 쓰기(보기에서 골라) (맞는 어휘를 보기에서 찾아 쓴다.), 12: 어휘 쓰기(본문에서 찾아) (맞는 어휘를 본문에서 찾아 쓴다.), 13: 어휘 쓰기(고쳐/직접) (맞는 어휘로 고쳐쓰거나 직접쓴다.), 14: 문장 쓰기 (문장을 쓴다.), 15: 바른 배열 쓰기 (맞는 배열하여 쓴다.), 16: 혼합, 17: 기타]"
            query_['input'] += f"\n<Method>정답<Method> 형태로 답하라. (예시, <Method>5<Method>)"
            query_['output'] = str(query_db['method'][idx])
    
        query_list.append(query_)
                                

    return query_list



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split',type=str, default="tr", choices=['tr', 'val' ,'te'])
    
    parser.add_argument('--vector_db_path', type=str, default="./data/vector_db.json")
    parser.add_argument('--query_path', type=str, default="./data/solvook_handout_te.csv")
    
    parser.add_argument('--chunk_size', type=int, default=8000)
    parser.add_argument('--chunk_overlap', type=int, default=200)
    
    parser.add_argument('--embedding_model', type=str, default="text-embedding-3-small")
    
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], default=1,)

    args = parser.parse_args()
    
    if args.task == 1:
        task_name = 'paragraph'
        K = 6
    elif args.task == 2:
        task_name = 'relation'
        K = 6
    elif args.task == 3:
        task_name = 'skill'
        K = 3
    elif args.task == 4:
        task_name = 'method'
        K = 3

    if args.split == 'tr':
        args.query_path = args.db_path
    
    ### [Step 1] Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
    print("[Step 1] Load Data!!")
    vector_db_dict, query_db = load_solvook_data(args)
    
    
    print("[Step 2] Start generation...")
    query_list = generation(args, vector_db_dict, query_db)
    
    with open(f'./{task_name}_top{K}_{args.split}.jsonl', 'w') as file:
        for query in query_list:
            file.write(json.dumps(query, ensure_ascii=False) + '\n')
    print("Finally End generation!!")
    