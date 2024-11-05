import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import os.path as osp


def remove_non_alphanumeric_prefix(input_string):
    # 문장의 시작이나 중간 또는 끝에 있는 문제 번호를 삭제하는 정규표현식 패턴
    pattern = r'^\d+\.\s*|\b\d+\b\)?\.?\s*'
    # 패턴에 맞는 부분을 찾아서 삭제
    result = re.sub(pattern, '', input_string)

    return result.strip()



def alphabet_to_number(char):
    char = char.lower()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if char in alphabet:
        return alphabet.index(char) + 1
    else:
        return None


def load_handout_db(data_path, handout_db, save=False):
    '''
    load and preprocess handout_db
    '''
    # load [1_handout_db] sheet
    # handout_db = pd.read_csv(osp.join(data_path, 'handout_db.csv'))
    handout_db = handout_db[['handout\nID', 'seq \n(교과서 코드)',
                        '분류체계 시트 > \n문제유형', '본문\n관계성_a',
                        '본문', '조건', '질문',  # '선지', '정답', '정답 해설',
                        'story\nID', 'paragraph\nunit', 'paragraph\nID_a', '교과서명']]
    handout_db.rename(columns={
                            "본문\n관계성_a" : "relation",
                            "handout\nID" : "handout_id",
                            "paragraph\nID_a" : "paragraph_id",
                            "paragraph\nunit" : "unit_id",
                            "story\nID" : "story_id",
                            "교과서명" : "textbook_id",
                            " 질문" : "질문"
                            }, inplace=True)

    handout_w_parap, handout_wo_parap = preprocess_handout_db(handout_db)

    if save:
        handout_w_parap.to_csv(osp.join(data_path, 'handout_w_parap_db_processed.csv'),
                        encoding="utf-8-sig", index=False) 
        handout_wo_parap.to_csv(osp.join(data_path, 'handout_wo_parap_db_processed.csv'),
                        encoding="utf-8-sig", index=False)
        
    return handout_w_parap, handout_wo_parap



def preprocess_handout_db(handout_db):
    ## preprocess handout db --------------------------------------------------------------------------------------------------
    print("1_handout_db preprocessing....")
    print(f"Data shape : {handout_db.shape} (Original)")
    
    # removing place holder obs & problematic obs
    miss_idx = np.logical_and(handout_db['handout_id'].isna(), handout_db['seq \n(교과서 코드)'].isna())
    handout_db = handout_db[~miss_idx]
    handout_db = handout_db.reset_index(drop=True)
    print(f"Data shape : {handout_db.shape} (Removed missing values in index)")
    
    miss_idx_ext = handout_db['분류체계 시트 > \n문제유형'].isna()
    handout_db = handout_db[~miss_idx_ext]
    handout_db = handout_db.reset_index(drop=True)
    print(f"Data shape : {handout_db.shape} (Removed missing values in skill and method)")
    
    # infilling paragraph of integrated questions
    handout_db['본문'] = handout_db['본문'].fillna(method='ffill')
    handout_db['본문'] = handout_db['본문'].str.replace('\n', ' ')

    # removing questions missing obs
    handout_db = handout_db[~handout_db['질문'].isna()]
    handout_db = handout_db.reset_index(drop=True)
    print(f"Data shape : {handout_db.shape} (Remove questions missing obs)")

    # replacement of duplicated strings
    for col in ['본문','질문','선지','조건','정답','정답해설']:
        try:
            handout_db[col]=handout_db[col].apply(lambda x: re.sub('___+','___',x))
            print(f'replace the long underlines of {col} to three "_"s ')
        except:
            pass

    # remove non alphanumeric prefix
    handout_db['질문'] = handout_db['질문'].apply(remove_non_alphanumeric_prefix)
    # split skill/method into each column
    for i in range(len(handout_db)):
        handout_db.loc[i,'skill'] =\
        int(handout_db.loc[i,'분류체계 시트 > \n문제유형'][:3])
        handout_db.loc[i,'method'] =\
        str(handout_db.loc[i,'분류체계 시트 > \n문제유형'][3])
        handout_db.loc[i, 'method'] = alphabet_to_number(handout_db.loc[i, 'method'])
    print(f"Data shape :{handout_db.shape} (Split skill and method into each column)")
    
    handout_db = handout_db[['handout_id', '본문', '질문', 'skill', 'method', 'relation',
                    'story_id', 'unit_id', 'paragraph_id', 'textbook_id']]
    
    ## Split w/ paragraph v.s. w/o pagraph
    handout_w_parap = handout_db[~handout_db['paragraph_id'].isna()]
    handout_wo_parap = handout_db[handout_db['paragraph_id'].isna()]
    print(f"Data shape :{handout_w_parap.shape} (handout db w/ paragraph_id)")
    print(f"Data shape :{handout_wo_parap.shape} (handout db w/o paragraph_id)")
    
    # remove string value in textbook/story/unit/paragraph_id (e.g. "S/L" in story_id (paragraph\nunit))
    # and change textbook/story/unit/paragraph_id into integer type
    handout_w_parap['unit_id'].replace('S/L', 0, inplace=True) ## hard coding
    for column in ['textbook_id', 'story_id', 'unit_id', 'paragraph_id', 'skill', 'method', 'relation']:
        handout_w_parap[column].fillna(0, inplace=True)
        handout_w_parap[column] = handout_w_parap[column].astype(int)
        # handout_db[column].replace(0, np.nan, inplace=True)
    
    handout_w_rel = handout_w_parap[~handout_w_parap['relation'].isna()]
    print(f"Data shape :{handout_w_rel.shape} (handout db w/ relation)")
    
    handout_wo_rel = handout_w_parap[handout_w_parap['relation'].isna()]
    print(f"Data shape :{handout_wo_rel.shape} (handout db w/o relation)")
        
    print("-"*30)
    
    return handout_w_parap, handout_wo_parap



def load_paragraph_db(data_path, paragraph_db, paragraph_list, save=False):
    '''
    load and preprocess paragraph_db
    '''    
    # load [2_paragraph_db] sheet
    paragraph_db = pd.read_csv(osp.join(data_path, 'paragraph_db.csv'))
    paragraph_db = paragraph_db[['textbook_id', 'unit_id', 'story_id', 'paragraph_id', 'paragraphs']]
    paragraph_db = preprocess_paragraph_db(paragraph_db)
    
    paragraph_list = pd.read_csv(osp.join(data_path, 'paragraph_list.csv'))
    paragraph_list = paragraph_list[['textbook_id', '교과서명',
                            'unit_id', 'unit_name',
                            'story_id_a', 'story name']]
    paragraph_list.rename(columns={'story_id_a' : 'story_id'}, inplace=True)
    paragraph_list.rename(columns={'교과서명' : 'textbook_name', 'story name' : 'story_name'}, inplace=True)
    paragraph_db = pd.merge(paragraph_db, paragraph_list, on=['textbook_id', 'unit_id', 'story_id'], how='left')
    print("Left outer join paragraph db with paragraph list db")
    print(f"Final Paragraph db shape : {paragraph_db.shape}")
    if save:
        paragraph_db.to_csv(osp.join(data_path, 'paragraph_db_processed.csv')
                        , encoding='utf-8-sig', index=False)
    return paragraph_db


def preprocess_paragraph_db(paragraph_db):
    print("2_paragraph_db preprocessing....")
    print(f"Data shape : {paragraph_db.shape} (Original)")
    miss_idx = np.logical_or(paragraph_db['textbook_id'].isna(), paragraph_db['unit_id'].isna())
    miss_idx = np.logical_or(miss_idx, paragraph_db['story_id'].isna())
    miss_idx = np.logical_or(miss_idx, paragraph_db['paragraph_id'].isna())
    miss_idx = np.logical_or(miss_idx, paragraph_db['paragraphs'].isna())
    paragraph_db = paragraph_db[~miss_idx]
    paragraph_db = paragraph_db.reset_index(drop=True)
    print(f"Data shape : {paragraph_db.shape} (Remove textbook, unit, story, paragraph id, and paragraph missing obs)")
    
    # replacement of duplicated strings
    try:
        paragraph_db['paragraphs']=paragraph_db['paragraphs'].apply(lambda x: re.sub('___+','___',x))
        print('replace the long underlines of paragraphs to three "_"s ')
    except:
        pass
    print("-"*30)
    
    return paragraph_db


def preprocess_paragraph_list(paragraph_list):
    print("4_paragraph_list preprocessing....")
    print(f"Data shape : {paragraph_list.shape} (Original)")
    
    miss_idx = np.logical_or(paragraph_list['textbook_id'].isna(), paragraph_list['unit_id'].isna())
    miss_idx = np.logical_or(miss_idx, paragraph_list['story_id'].isna())
    paragraph_list = paragraph_list[~miss_idx]
    paragraph_list = paragraph_list.reset_index(drop=True)
    print(f"Data shape : {paragraph_list.shape} (Remove textbook, unit, and story id missing obs)")
    
    miss_idx = paragraph_list['교과서명'].isna()
    miss_idx = np.logical_or(miss_idx, paragraph_list['unit_name'].isna())
    miss_idx = np.logical_or(miss_idx, paragraph_list['story_name'].isna())
    paragraph_list = paragraph_list[~miss_idx]
    paragraph_list = paragraph_list.reset_index(drop=True)
    print(f"Data shape : {paragraph_list.shape} (Remove 교과서명, textbook, unit, and, story name missing obs)")
    
    print("-"*30)
    return paragraph_list




def get_dataset_split(handout_db,
                      paragraph_db,
                      ratio=[0.8,0.1,0.1],
                      split_random_seed=0):
    assert sum(ratio) == 1, 'sum of [tr,val,te] proportions must be 1'
    r_valte = sum(ratio[1:])
    r_te = ratio[-1] / r_valte

    # tr / te 
    tr, te_0 = train_test_split(handout_db, 
                                stratify=handout_db['relation'],
                                test_size=r_valte,
                                random_state=split_random_seed)
    # val / te
    val, te  = train_test_split(te_0,
                                stratify=te_0['relation'],
                                test_size=r_te,
                                random_state=split_random_seed)

    # Join training handout_db with paragraph_db
    tr = pd.merge(tr, paragraph_db, on=['textbook_id', 'unit_id', 'story_id', 'paragraph_id'], how='left').reset_index(drop=True)
    val = pd.merge(val, paragraph_db, on=['textbook_id', 'unit_id', 'story_id', 'paragraph_id'], how='left').reset_index(drop=True)
    te = pd.merge(te, paragraph_db, on=['textbook_id', 'unit_id', 'story_id', 'paragraph_id'], how='left').reset_index(drop=True)

    return tr, val, te



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path',type=str, default='/data1/lsj9862/tips_2024/data')
    
    args = parser.parse_args()
    
    db = pd.read_excel('/data1/lsj9862/tips_2024/data/Solvook_handout_DB_english.xlsx', sheet_name=None)
    handout_db = db['1_handout_db']
    paragraph_db = db['2_paragraph_db']
    paragraph_list = db['4_paragraph_list']
    
    # load handout_db, paragraph_db
    handout_w_parap, handout_wo_parap = load_handout_db(data_path = args.data_path,
                                                    handout_db = handout_db)
    paragraph_db = load_paragraph_db(data_path = args.data_path,
                                    paragraph_db = paragraph_db,
                                    paragraph_list = paragraph_list)
    
    # split tr (document) / val / test in "handout with paragraph" db
    tr_set, val_set, te_set = get_dataset_split(handout_db=handout_w_parap,
                            paragraph_db=paragraph_db,
                            ratio=[0.8,0.1,0.1],
                            split_random_seed=args.seed)
    
    # add "handout w/o paragraph" into tr (document) set
    tr_set = pd.concat((tr_set, handout_wo_parap))
    
    print(f'train split: {tr_set.shape}')
    print(f'validation split: {val_set.shape}')
    print(f'test split: {te_set.shape}')
    print("-"*30)
    
    tr_set.to_csv(osp.join(args.data_path,'solvook_handout_tr.csv'), index=False, encoding='utf-8-sig')
    val_set.to_csv(osp.join(args.data_path,'solvook_handout_val.csv'), index=False, encoding='utf-8-sig')
    te_set.to_csv(osp.join(args.data_path,'solvook_handout_te.csv'),  index=False, encoding='utf-8-sig')
    
    print("Complete preprocessing!!")