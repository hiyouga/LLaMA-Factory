import pandas as pd
import numpy as np
import json
import os
import re
from tqdm import tqdm

# 1. Prepare raw-data
def prepare_raw_data(data_path):
    """
    Load and prepare the raw data from the Excel file.
    """
    print("Loading raw data from Excel file...")
    all_sheet = pd.read_excel(data_path, sheet_name=None)
    handout_db = all_sheet['1_handout_db']
    paragraph_db = all_sheet['2_paragraph_db']
    paragraph_list = all_sheet['4_paragraph_list']
    contents_type_sys = all_sheet['Contents_분류체계ver2.0']

    handout_db = handout_db.dropna(subset=['질문'])
    selected_columns = ["handout\nID", "_x0008_1지문\n다문항", "본문", "조건", "선지", "정답", "질문", 
                        "분류체계 시트 > \n문제유형", "분류체계 시트 > \nskill # , method", "unit_name", 
                        "story_name", "교과서명"]
    new_df = handout_db[selected_columns].reset_index(drop=True)
    new_column_names = {
        'handout\nID': 'handoutID',
        '분류체계 시트 > \n문제유형': '문제유형',
        '분류체계 시트 > \nskill # , method': 's&m',
        '_x0008_1지문\n다문항': '1지문다문항'
    }
    new_df = new_df.rename(columns=new_column_names)
    
    for idx in range(len(new_df)):
        if (new_df['1지문다문항'][idx] > 0) and (str(new_df['본문'][idx]) == 'nan'):
            new_df.loc[idx, '본문'] = new_df.loc[idx - 1, '본문']

    return new_df, paragraph_db, paragraph_list, contents_type_sys

# 2. Preprocessing Question column
def preprocess_question_columns(df):
    """
    Preprocess the question column to clean up formatting.
    """
    print("Preprocessing question columns...")
    def remove_non_alphanumeric_prefix(input_string):
        pattern = '[가-힣a-zA-Z]'
        match = re.search(pattern, input_string)
        if match:
            cleaned_string = input_string[match.start():]
        else:
            cleaned_string = input_string
        return cleaned_string

    def remove_number_and_bracket(text):
        return re.sub(r'\s*\d+\)', '', text)

    for col in ['본문', '질문', '선지', '조건', '정답']:
        for idx in range(len(df['질문'])):
            try:
                df.loc[idx, col] = re.sub('___+', '___', df.loc[idx, col])
            except:
                pass

    df['질문'] = df['질문'].apply(remove_number_and_bracket).apply(remove_non_alphanumeric_prefix)
    return df

# 3. Preprocessing s&m column
def preprocess_skill_method_column(df):
    """
    Extract skill and method from the 's&m' column and preprocess the dataframe.
    """
    print("Preprocessing skill and method columns...")
    def extract_skill_method(s_and_m):
        try:
            s_and_m_dict = json.loads(s_and_m.replace("[", "\"").replace("]", "\""))
            skill = s_and_m_dict.get('skill', None)
            method = s_and_m_dict.get('method', None)
            return skill, method
        except:
            return None, None

    df[['skill', 'method']] = df['s&m'].apply(extract_skill_method).apply(pd.Series)
    df['s&m'] = df['s&m'].str.replace('[', '"').str.replace(']', '"')
    df.rename(columns={'unit_name': 'unit_id', '교과서명': 'textbook_id', 'story_name': 'story_name_id'}, inplace=True)
    return df

# 4. skill & method Dictionary
def create_skill_method_dicts(contents_type_sys):
    """
    Create dictionaries for skill, method, and quiz type from the given contents system.
    """
    print("Creating skill, method, and quiz type dictionaries...")
    skill_df = contents_type_sys.iloc[9:40, [1, 2, 5]]
    skill_dict = skill_df.set_index(skill_df.columns[0]).T.to_dict('list')
    skill_dict = {str(key): value for key, value in skill_dict.items()}
    
    method_df = contents_type_sys.iloc[44:61, [1, 2, 5]]
    method_dict = method_df.set_index(method_df.columns[0]).T.to_dict('list')
    method_dict = {str(key): value for key, value in method_dict.items()}
    
    quiztype_df = contents_type_sys.iloc[64:293, [1, 6]]
    quiztype_df = quiztype_df[quiztype_df['참조 및 근거 출처'] != '-']
    quiztype_dict = quiztype_df.set_index(quiztype_df.columns[0]).T.to_dict('list')

    return skill_dict, method_dict, quiztype_dict

# 5. Filtering Data
def filter_handout_data(df, quiztype_dict):
    """
    Filter the handout data to include only valid quiz types.
    """
    print("Filtering handout data based on quiz types...")
    return df[df['문제유형'].isin(list(quiztype_dict.keys()))].copy().reset_index(drop=True)

# 6. Merging sheets
def merge_paragraph_data(pp_handout, paragraph_db, paragraph_list):
    """
    Merge paragraph data based on textbook_id and unit_id.
    """
    print("Merging paragraph data...")
    grouped_paragraph_db = paragraph_db.groupby(['textbook_id', 'unit_id'], as_index=False)['paragraphs'].apply(', '.join)
    pp_paragraph = pd.merge(grouped_paragraph_db, paragraph_list[['textbook_id', 'unit_id', '교과서명', '출판사', 'unit_title', 'unit_name', 'story name', 'story type']], on=['textbook_id', 'unit_id'])

    def extract_unit_textbook(df):
        try:
            unit = df['unit_id']
            textbook = df['textbook_id']
            return (int(unit), int(textbook))
        except (ValueError, TypeError):
            return (None, None)

    pp_handout['u&t'] = pp_handout.apply(extract_unit_textbook, axis=1)
    pp_paragraph['u&t'] = pp_paragraph.apply(extract_unit_textbook, axis=1)
    
    connection_ = pp_paragraph.drop(['textbook_id', 'unit_id', 'paragraphs', '출판사', 'unit_title', 'story type'], axis=1)
    pp_connection = pd.merge(pp_handout, connection_, how='left', on='u&t')
    
    return pp_connection, pp_paragraph

# 7. Generate Equal/ Difference Quiztype Set
def generate_equal_diff_pairs(pp_handout):
    """
    Generate question pairs where quiz types are either equal or different.
    """
    print("Generating equal and different quiz type question pairs...")
    equal_pair_query, diff_pair_query = [], []
    for idx in tqdm(range(len(pp_handout))):
        question, quiztype = pp_handout['질문'][idx], pp_handout['문제유형'][idx]
        same_quiztype = pp_handout[pp_handout['문제유형'] == quiztype]
        diff_quiztype = pp_handout[pp_handout['문제유형'] != quiztype]
        if len(same_quiztype) != 1:
            equal_pair_query.append([question, same_quiztype[same_quiztype['질문'] != question]['질문'].sample(n=1, random_state=42).values[0]])
            diff_pair_query.append([question, diff_quiztype['질문'].sample(n=1, random_state=42).values[0]])

    return equal_pair_query, diff_pair_query

# 8. Generate instruction prompts
def generate_instruction_prompts(pp_handout, skill_dict, method_dict, quiztype_dict, pp_connection, equal_pair_query, diff_pair_query, pp_paragraph):
    """
    Generates instruction prompts based on the handout and other data.
    """
    print("Generating instruction prompts...")
    prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10 = [], [], [], [], [], [], [], [], [], []

    prompt_style = {
        "instruction": "instruction: 다음은 문제를 설명하는 지침입니다. 지침에 따라 문제를 적절하게 완료하는 응답을 작성하십시오. \n{instruction}",
        "input": "input:{input}",
        "output": "output:{output}",
        "handoutID": "handoutID:{handoutID}"
    }

    for idx in range(len(pp_handout['질문'])):
        query = pp_handout.loc[idx, '질문']
        ID = pp_handout.loc[idx, 'handoutID']
        passage = pp_handout.loc[idx, '본문'] if pd.notnull(pp_handout.loc[idx, '본문']) else '없음'
        condition = pp_handout.loc[idx, '조건'] if pd.notnull(pp_handout.loc[idx, '조건']) else '없음'
        choices = pp_handout.loc[idx, '선지'] if pd.notnull(pp_handout.loc[idx, '선지']) else '없음'
        answer = pp_handout.loc[idx, '정답'] if pd.notnull(pp_handout.loc[idx, '정답']) else False
        
        try:
            quiztype = quiztype_dict[pp_handout.loc[idx, '문제유형']][0]
        except:
            quiztype = False
        
        skill_index = pp_handout.loc[idx, 'skill']
        method_index = pp_handout.loc[idx, 'method']
        sm = 'skill: ' + skill_dict[skill_index][0] + ', method: ' + method_dict[method_index][0]
        s = 'skill: ' + skill_dict[skill_index][0]
        m = 'method: ' + method_dict[method_index][0]

        # Instruction 1: 질문 유형과 관련된 skill, method 답하기
        if quiztype:
            instruction1 = "다음 질문이 어떤 문제 유형에 해당하는지, 고등 영어 교육과정 중 어떤 skill, method과 관련 있는지 답하라."
            input1 = f"질문 : {query}\n조건 : {condition}\n선지 : {choices}"
            output1 = f"'{s}' \n '{m}'"
            prompt1.append({
                'instruction': prompt_style['instruction'].format(instruction=instruction1),
                'input': prompt_style['input'].format(input=input1),
                'output': prompt_style['output'].format(output=output1),
                'handoutID': prompt_style['handoutID'].format(handoutID=ID)
            })

        # Instruction 2: 질문을 풀기 위해 필요한 method 답하기
        instruction2 = "다음 질문에 답하기 위해서는 어떤 method가 필요한가?"
        input2 = f"method 후보: {[value[0] for value in method_dict.values()]}, \n질문: {query}"
        output2 = f"'{m}'를 알면 질문을 풀 수 있다"
        prompt2.append({
            'instruction': prompt_style['instruction'].format(instruction=instruction2),
            'input': prompt_style['input'].format(input=input2),
            'output': prompt_style['output'].format(output=output2),
            'handoutID': prompt_style['handoutID'].format(handoutID=ID)
        })

        # Instruction 3: 질문을 풀기 위해 필요한 skill 답하기
        instruction3 = "다음 질문에 답하기 위해서는 어떤 skill이 필요한가?"
        input3 = f"skill 후보: {[value[0] for value in skill_dict.values()]}, \n질문: {query}"
        output3 = f"제시된 질문에 답을 하기 위해 '{s}'를 알아야 한다"
        prompt3.append({
            'instruction': prompt_style['instruction'].format(instruction=instruction3),
            'input': prompt_style['input'].format(input=input3),
            'output': prompt_style['output'].format(output=output3),
            'handoutID': prompt_style['handoutID'].format(handoutID=ID)
        })

        # Instruction 4: 지문과 관련된 문제 유형 답하기
        if quiztype:
            instruction4 = "다음 지문에 대해 출제된 질문의 문제유형은 무엇인가?"
            input4 = f"문제유형 후보: {[value[0] for value in quiztype_dict.values()]}, \n지문: {passage}, \n질문: {query}"
            output4 = f"'{quiztype}'이다."
            prompt4.append({
                'instruction': prompt_style['instruction'].format(instruction=instruction4),
                'input': prompt_style['input'].format(input=input4),
                'output': prompt_style['output'].format(output=output4),
                'handoutID': prompt_style['handoutID'].format(handoutID=ID)
            })

        # Instruction 5: 정답 맞추기
        if answer:
            instruction5 = "제시된 문제의 정답을 맞춰라."
            input5 = f"지문: {passage}, \n조건: {condition}, \n질문: {query}, \n선지: {choices}"
            output5 = f"문제의 정답은 '{answer}'이다."
            prompt5.append({
                'instruction': prompt_style['instruction'].format(instruction=instruction5),
                'input': prompt_style['input'].format(input=input5),
                'output': prompt_style['output'].format(output=output5),
                'handoutID': prompt_style['handoutID'].format(handoutID=ID)
            })

        # Instruction 6: 지문에 어울리는 질문 찾기
        instruction6 = "다음과 같은 형태의 지문에 출제할 수 있는 질문은 무엇인가?"
        if passage != '없음':
            input6 = f"지문: {passage}"
            output6 = f"'{query}'이다."
            prompt6.append({
                'instruction': prompt_style['instruction'].format(instruction=instruction6),
                'input': prompt_style['input'].format(input=input6),
                'output': prompt_style['output'].format(output=output6),
                'handoutID': prompt_style['handoutID'].format(handoutID=ID)
            })

        # Instruction 10: 지문과 교과서의 연관성 찾기
        if not pd.isnull(pp_connection.loc[idx, 'unit_name']):
            paragraph_connection_list = ['unit_name', 'story name', '교과서명']
            paragraph_connection = []
            for connect in paragraph_connection_list:
                paragraph_connection.append(str(pp_connection.loc[idx, connect]))
            paragraph_connection = '-'.join(paragraph_connection)

            instruction10 = "제시된 지문이 어떤 교과서의 본문과 연관되는지 답하여라."
            input10 = f"지문: {passage}"
            output10 = f"주어진 지문과 연관되어 있는 교과서는 {paragraph_connection}이다."
            prompt10.append({
                'instruction': prompt_style['instruction'].format(instruction=instruction10),
                'input': prompt_style['input'].format(input=input10),
                'output': prompt_style['output'].format(output=output10),
                'handoutID': prompt_style['handoutID'].format(handoutID=ID)
            })

    # Additional prompts for equal and different pairs (Instruction 7)
    for sim_idx in range(len(equal_pair_query)):
        equal_querys = equal_pair_query[sim_idx]
        
        instruction7_1 = "다음 두 질문의 문제 유형이 같은지 다른지 답하라."
        input7_1 = f"질문1: {equal_querys[0]}, 질문2: {equal_querys[1]}"
        output7_1 = "두 질문의 문제 유형이 같다"
        prompt7.append({
            'instruction': prompt_style['instruction'].format(instruction=instruction7_1),
            'input': prompt_style['input'].format(input=input7_1),
            'output': prompt_style['output'].format(output=output7_1),
            'handoutID': prompt_style['handoutID'].format(handoutID=ID)
        })

    for sim_idx in range(len(diff_pair_query)):
        diff_querys = diff_pair_query[sim_idx]

        instruction7_2 = "다음 두 질문의 문제 유형이 같은지 다른지 답하라."
        input7_2 = f"질문1: {diff_querys[0]}, 질문2: {diff_querys[1]}"
        output7_2 = "두 질문의 문제 유형이 다르다"
        prompt7.append({
            'instruction': prompt_style['instruction'].format(instruction=instruction7_2),
            'input': prompt_style['input'].format(input=input7_2),
            'output': prompt_style['output'].format(output=output7_2),
            'handoutID': prompt_style['handoutID'].format(handoutID=ID)
        })

    # Instruction 8 and 9: 글 종류와 제목
    for par_idx in range(len(pp_paragraph['paragraphs'])):
        paragraph = pp_paragraph.loc[par_idx, 'paragraphs']
        story_name = pp_paragraph.loc[par_idx, 'story name']
        story_type = pp_paragraph.loc[par_idx, 'story type']

        paragraph_info_list = ['교과서명', '출판사', 'unit_title', 'unit_name']
        paragraph_info = []
        for para in paragraph_info_list:
            paragraph_info.append(pp_paragraph.loc[par_idx, para])
        paragraph_info = '-'.join(paragraph_info)

        instruction8 = f"제시된 본문은 '{paragraph_info}'에 해당한다. 본문의 글의 종류는 무엇인가?"
        input8 = f"본문: {paragraph}"
        output8 = f"글의 종류는 '{story_type}'이다."
        prompt8.append({
            'instruction': prompt_style['instruction'].format(instruction=instruction8),
            'input': prompt_style['input'].format(input=input8),
            'output': prompt_style['output'].format(output=output8),
            'handoutID': prompt_style['handoutID'].format(handoutID=ID)
        })

        instruction9 = f"제시된 본문은 '{paragraph_info}'에 포함된다. 본문의 제목은 무엇인가?"
        input9 = f"본문: {paragraph}"
        output9 = f"글의 제목은 '{story_name}'이다."
        prompt9.append({
            'instruction': prompt_style['instruction'].format(instruction=instruction9),
            'input': prompt_style['input'].format(input=input9),
            'output': prompt_style['output'].format(output=output9),
            'handoutID': prompt_style['handoutID'].format(handoutID=ID)
        })

    return [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]


# 9. Save JSON
def save_json(prompts, filename='example.json', folder='Instruction_dataset'):
    """
    Save the generated instruction prompts to a JSON file inside the specified folder.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Set file path inside the folder
    filepath = os.path.join(folder, filename)
    
    print(f"Saving prompts to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', default='./data')
    args = parser.parse_args()
    
    print("Starting instruction tuning data preparation...")

    # Set working directory
    os.chdir(args.working_dir)

    # Prepare raw data
    data_path = os.path.join(args.working_dir, 'Solvook_handout_DB_english.xlsx')
    new_df, paragraph_db, paragraph_list, contents_type_sys = prepare_raw_data(data_path)

    # Preprocess columns
    new_df = preprocess_question_columns(new_df)
    new_df = preprocess_skill_method_column(new_df)

    # Create dictionaries
    skill_dict, method_dict, quiztype_dict = create_skill_method_dicts(contents_type_sys)

    # Filter handout dataa
    pp_handout = filter_handout_data(new_df, quiztype_dict)

    # Merge paragraph data
    pp_connection, pp_paragraph = merge_paragraph_data(pp_handout, paragraph_db, paragraph_list)

    # Generate pairs
    equal_pair_query, diff_pair_query = generate_equal_diff_pairs(pp_handout)

    # Generate instruction prompts
    promptlist = generate_instruction_prompts(pp_handout, skill_dict, method_dict, quiztype_dict, pp_connection, equal_pair_query, diff_pair_query, pp_paragraph)

    # Save prompts in the Instruction_dataset folder
    for idx, prompts in enumerate(promptlist):
        save_json(prompts, f'example{idx+1}.json')

    # Save all prompts together
    all_prompts = [item for sublist in promptlist for item in sublist]
    save_json(all_prompts, 'Solvook_instruction.json')

    print("Instruction tuning dataset preparation completed.")

