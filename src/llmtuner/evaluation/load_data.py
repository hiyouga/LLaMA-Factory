import json
import os
import random

def load_mCOPA(preprocess,template):  
    
    if preprocess == 'zero_shot_xcopa':
        f = open("/mnt/data/shesj/LLMEval_data/xcopa/xcopa_zero_shot.json",'r')
        test_samples = json.load(f)
        print(test_samples[-1])
        return test_samples
    if preprocess == 'few_shot_xcopa':
        f = open("/mnt/data/shesj/LLMEval_data/xcopa/xcopa_four_shot.json",'r')
        test_samples = json.load(f)
        print(test_samples[-1])
        return test_samples


def load_mMMLU(preprocess,template):
    mmlu_base_dir = "/mnt/data/shesj/LLMEval_data/mMMLU/"
    language_map = {
        'bn' : "Bengali",
        'en' : "English",
        'de' : "German",
        "es" : "Spanish",
        "fr" : "French",
        "ru" : "Russian",
        "zh" : "Chinese",
        "ja": "Japanese",  # Fixed typo
        "th": "Thai",      # Updated for consistency
        "sw" : "Swahili"
    }

    system_prompt_map = {
        "English": "The following are multiple choice questions (with answers). Please think step by step and choose the best option.",
        "Bengali": "নিম্নলিখিতগুলি একাধিক পছন্দের প্রশ্ন (উত্তরসহ)। দয়া করে ধাপে ধাপে চিন্তা করুন এবং সেরা বিকল্পটি চয়ন করুন।",
        "German": "Die folgenden sind Multiple-Choice-Fragen (mit Antworten). Bitte denken Sie Schritt für Schritt nach und wählen Sie die beste Option.",
        "Spanish": "A continuación se presentan preguntas de opción múltiple (con respuestas). Por favor, piense paso a paso y elija la mejor opción.",
        "French": "Ce qui suit sont des questions à choix multiples (avec réponses). Veuillez réfléchir étape par étape et choisir la meilleure option.",
        "Russian": "Ниже приведены вопросы с выбором ответа (с ответами). Пожалуйста, подумайте пошагово и выберите лучший вариант.",
        "Chinese": "以下是多项选择题（附带答案）。请逐步思考并选择最佳选项。",
        "Japanese": "以下は複数選択の質問（解答付き）です。ステップバイステップで考え、最適な選択肢を選んでください。",
        "Thai": "ต่อไปนี้เป็นคำถามแบบเลือกตอบ (พร้อมคำตอบ) โปรดคิดทีละขั้นตอนและเลือกตัวเลือกที่ดีที่สุด",
        "Swahili": "Yafuatayo ni maswali ya kuchagua mara nyingi (yenye majibu). Tafadhali fikiria hatua kwa hatua na uchague chaguo bora."
    }

    question_answer_template_map = {
        "English": "\n\nQuestion: {}\n(A): {} (B): {} (C): {} (D): {}\nAnswer: Let's think step by step. {} The answer is ({}).",
        "Bengali": "\n\nপ্রশ্ন: {}\n(A): {} (B): {} (C): {} (D): {}\nউত্তর: আসুন ধাপে ধাপে চিন্তা করি. {} উত্তরটি হল ({}).",
        "German": "\n\nFrage: {}\n(A): {} (B): {} (C): {} (D): {}\nAntwort: Lassen Sie uns Schritt für Schritt denken. {} Die Antwort ist ({}).",
        "Spanish": "\n\nPregunta: {}\n(A): {} (B): {} (C): {} (D): {}\nRespuesta: Pensemos paso a paso. {} La respuesta es ({}).",
        "French": "\n\nQuestion: {}\n(A): {} (B): {} (C): {} (D): {}\nRéponse: Réfléchissons étape par étape. {} La réponse est ({}).",
        "Russian": "\n\nВопрос: {}\n(A): {} (B): {} (C): {} (D): {}\nОтвет: Давайте думать шаг за шагом. {} Ответ - ({}).",
        "Chinese": "\n\n问题: {}\n(A): {} (B): {} (C): {} (D): {}\n答案: 让我们一步一步来思考。{} 答案是 ({}).",
        "Japanese": "\n\n質問: {}\n(A): {} (B): {} (C): {} (D): {}\n回答: ステップバイステップで考えてみましょう。 {} 答えは ({}).",
        "Thai": "\n\nคำถาม: {}\n(A): {} (B): {} (C): {} (D): {}\nคำตอบ: มาคิดกันทีละขั้นตอน {} คำตอบคือ ({}).",
        "Swahili": "\n\nSwali: {}\n(A): {} (B): {} (C): {} (D): {}\nJibu: Hebu tufikirie hatua kwa hatua. {} Jibu ni ({})."
    }

    question_template_map = {
        "English": "\n\nQuestion: {}\n(A): {} (B): {} (C): {} (D): {}\nAnswer: Let's think step by step.",
        "Bengali": "\n\nপ্রশ্ন: {}\n(A): {} (B): {} (C): {} (D): {}\nউত্তর: আসুন ধাপে ধাপে চিন্তা করি.",
        "German": "\n\nFrage: {}\n(A): {} (B): {} (C): {} (D): {}\nAntwort: Lassen Sie uns Schritt für Schritt denken.",
        "Spanish": "\n\nPregunta: {}\n(A): {} (B): {} (C): {} (D): {}\nRespuesta: Pensemos paso a paso.",
        "French": "\n\nQuestion: {}\n(A): {} (B): {} (C): {} (D): {}\nRéponse: Réfléchissons étape par étape.",
        "Russian": "\n\nВопрос: {}\n(A): {} (B): {} (C): {} (D): {}\nОтвет: Давайте думать шаг за шагом.",
        "Chinese": "\n\n问题: {}\n(A): {} (B): {} (C): {} (D): {}\n答案: 让我们一步一步来思考。",
        "Japanese": "\n\n質問: {}\n(A): {} (B): {} (C): {} (D): {}\n回答: ステップバイステップで考えてみましょう。",
        "Thai": "\n\nคำถาม: {}\n(A): {} (B): {} (C): {} (D): {}\nคำตอบ: มาคิดกันทีละขั้นตอน",
        "Swahili": "\n\nSwali: {}\n(A): {} (B): {} (C): {} (D): {}\nJibu: Hebu tufikirie hatua kwa hatua."
    }

    def load_data(base_path):
        print(base_path)
        multilingual_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request in {lang}. Please answer in {lang}.\n\n### Instruction:\n{instruction}\n\n### Response:"""
        
        lang = base_path.split('/')[-1]
        lang = language_map[lang]
        file_path = base_path + "/example.jsonl"
        f = open(file_path)
        
        samples = json.load(f)

        map2example = {}
        for subject in samples:
            if 'zero_shot_mmmlu' == preprocess:
                example = samples[subject][:0]
            else:
                example = samples[subject][:2]

            #system_prompt = "The following are multiple choice questions (with answers)."
            system_prompt = system_prompt_map[lang]
            for e in example:
                system_prompt += question_answer_template_map[lang].format(e['instruction'] ,e['option_a'] ,e['option_b'],  e['option_c'],  e['option_d'],e['reasoning'],e['answer'])
            
            map2example[subject] = system_prompt

        file_path = base_path + "/test.jsonl"
        f = open(file_path)
        lines = f.readlines()
        test_data = []

        for i in lines:
            d = json.loads(i.strip())
            if 'id' not in d:
                print(d)
            subject = d['id'].split('/')[0]
            #print(subject)
            cur_sys_prompt = map2example[subject]
            temp = cur_sys_prompt + question_template_map[lang].format(d['instruction'] , d['option_a'] , d['option_b'],  d['option_c'], d['option_d'])
            d['lang'] = lang
            d['prompted'] = multilingual_template.format(lang = lang,instruction = temp)
            test_data.append(d)
        return test_data

    file = os.listdir(mmlu_base_dir)


    test_samples = []


    id_counter = {}
    for i in file:
        if i == ".DS_Store":
            continue

        cur_lang_data = load_data(mmlu_base_dir + i)
        for c in cur_lang_data:
            id_counter[c['id']] = id_counter.get(c['id'],0) + 1
    
    good_id = []
    for id in id_counter:
        if id_counter[id] == 10:
            good_id.append(id)

    print(len(id_counter))
    print(len(good_id))
    random.seed(42)
    indices = random.sample(range(len(good_id)), 200)
    selected_id = [good_id[idx] for idx in indices]


    for i in file:
        if i == ".DS_Store":
            continue

        selected_data = []
        cur_lang_data = load_data(mmlu_base_dir + i)
        for c in cur_lang_data:
            if c['id'] in selected_id:
                selected_data.append(c)

        print(len(selected_data))

        test_samples += selected_data


        if i=='zh':
            for i in selected_data[:5]:
                print(i['prompted'])
                print("\n\n")
    #print(test_samples[-10])
    # exit()
    return test_samples
        


def load_MMLU(preprocess,template):
    mmlu_base_dir = "/mnt/data/shesj/LLMEval_data/mmlu/"
    file = os.listdir(mmlu_base_dir)
    
    def load_example(base_path):
        test_samples = []
        f = open(base_path)
        train_lines = f.readlines()
        for i in train_lines:
            sample = json.loads(i.strip())
            sample['prompted'] = preprocess(sample['question'],template)
            sample['object'] = base_path.split("/")[-1]
            test_samples.append(sample)
        return test_samples

    all_sample = []
    for i in file:
        all_sample += load_example(mmlu_base_dir + i)
    
    print(all_sample[0]['prompted'])
    print(len(all_sample))
    # exit()
    return all_sample


def load_bbh_mc(preprocess,template):
    bbh_base_dir = "/mnt/data/shesj/LLMEval_data/bbh_mc_orca/"
    file = os.listdir(bbh_base_dir)
    def load_example(base_path):
        test_samples = []
        f = open(base_path)
        train_lines = f.readlines()
        for i in train_lines:
            sample = json.loads(i.strip())
            sample['prompted'] = preprocess(sample['question'],template)
            sample['object'] = base_path.split("/")[-1]
            test_samples.append(sample)
        return test_samples
    all_sample = []
    for i in file:
        all_sample += load_example(bbh_base_dir + i)
    
    print(all_sample[0]['prompted'])
    print(len(all_sample))

    return all_sample


def load_bbh(preprocess,template):
    bbh_base_dir = "/mnt/data/shesj/LLMEval_data/bbh/"
    file = os.listdir(bbh_base_dir)
    def load_example(base_path):
        test_samples = []
        f = open(base_path)
        train_lines = f.readlines()
        for i in train_lines:
            sample = json.loads(i.strip())
            sample['prompted'] = preprocess(sample['question'],template)
            sample['object'] = base_path.split("/")[-1]
            test_samples.append(sample)
        return test_samples
    
            

    all_sample = []
    for i in file:
        all_sample += load_example(bbh_base_dir + i)
    
    print(all_sample[0]['prompted'])
    print(len(all_sample))
    return all_sample







def load_mSafe(preprocess,template):
    mmlu_base_dir = "/mnt/data/shesj/LLMEval_data/safe/data.json"
    language_map = {
        'bn' : "Bengali",
        'en' : "English",
        'de' : "German",
        "es" : "Spanish",
        "fr" : "French",
        "ru" : "Russian",
        "zh" : "Chinese",
        "ja" : "Janpanese",
        "th" : "Thailand",
        "sw" : "Swahili"
    }
    test_samples = []
    data = json.load(open(mmlu_base_dir,'r'))
    for i in data:
        multilingual_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request in {lang}. Please answer in {lang}.\n\n### Instruction:\n{instruction}\n\n### Response:"""
        lang = i['lang']
        instruction = i['instruction']
        d = {}
        d['lang'] = lang
        d['prompted'] = multilingual_template.format(lang = lang,instruction = instruction)
        test_samples.append(d)
    
    print(test_samples[-1])
    return test_samples
        

if __name__ == "__main__":
    # 只有当文件被直接运行时，这里的代码才会执行
    from enginee_utils import return_preprocess_function
    process_func = return_preprocess_function('raw')
    load_mMMLU(process_func,'none')
    print("------------------------------------------------")
    process_func = return_preprocess_function('single_turn_mmlu')
    load_MMLU(process_func,'vicuna_v1.1')
    print("------------------------------------------------")
    process_func = return_preprocess_function('single_turn_mmlu')
    load_MMLU(process_func,'NanGPT')
    print("------------------------------------------------")