

from conversation_utils import get_conv_template,SeparatorStyle


def preprocess_raw(input_string,template_name = "NanGPT"):
    conv = get_conv_template(template_name)
    conv.append_message(conv.roles[0], input_string)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()
    return prompted

def preprocess_zero_shot_bbh_mc(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)

    input_prompt = question_prompt + " Choose an answer from the options provided. At the end output The answer is: {answer choice}. "
    questions = questions[-1]
    input_part = "\n\nQuestion: " +  questions.split("A:")[0].strip()
    input_prompt += input_part
    # reasoning_part = "\nAnswers:"
    # input_prompt +=  reasoning_part

    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()  + " Let me think step by step." 
    # print(prompted)
    # exit()
    return prompted
    # print(prompted)

def preprocess_zero_shot_bbh(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    input_prompt = question_prompt + " Please solve it step by step carefully."
    questions = questions[-1]
    input_part = "\n\nQuestion: " +  questions.split("A: Let\'s think step by step.")[0].strip()
    input_prompt += input_part
    # reasoning_part = "\nAnswers:"
    # input_prompt +=  reasoning_part

    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()  + " Let me think step by step." 
    # print(prompted)
    # exit()

    return prompted

def preprocess_single_turn_bbh(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    input_prompt = question_prompt + " Please solve it step by step carefully."
    
    for q in questions:
        input_part = "\n\nQuestion: " +  q.split("A: Let\'s think step by step.")[0].strip()
        input_prompt += input_part
        reasoning_part = "\nAnswers: Let me think step by step. " + q.split("A: Let\'s think step by step.")[1].strip()
        input_prompt +=  reasoning_part

    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()
    # print(prompted)
    # exit()

    return prompted

def preprocess_few_shot_AO_bbh(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    input_prompt = question_prompt
    
    for q in questions:
        input_part = "\n\nQuestion: " +  q.split("A: Let\'s think step by step.")[0].strip()
        input_prompt += input_part
        if "the answer is " in q:
            reasoning_part = "\nAnswers: "  + q.split("A: Let\'s think step by step.")[1].strip().split("the answer is ")[1]
        else:
            reasoning_part = "\nAnswers: " + q.split("A: Let\'s think step by step.")[1].strip()
        input_prompt +=  reasoning_part

    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()

    return prompted

def preprocess_multi_turn_mmlu(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    #print(question_prompt)
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    conv.append_message(conv.roles[0], question_prompt + " Please solve it step by step carefully.")
    conv.append_message(conv.roles[1], "Of course! Please provide me with the multiple-choice questions you would like assistance with. As an expert of {}, I'll walk you through the solutions step by step.")
    
    for q in questions[:-1]:
        input_part = q.split("\nA: Let's think step by step. ")[0]
        conv.append_message(conv.roles[0], "Question: " + input_part)
        reasoning_part = q.split("\nA: Let's think step by step. ")[1]
        conv.append_message(conv.roles[1], "Let me think step by step. " + reasoning_part)

    q = questions[-1]
    last_question = q.split("\n\n")[0]
    target_question = q.split("\n\n")[1].replace("A: Let's think step by step.","").strip()

    input_part = last_question.split("\nA: Let's think step by step. ")[0]
    conv.append_message(conv.roles[0],  "Question: " + input_part)
    reasoning_part = last_question.split("\nA: Let's think step by step. ")[1]
    conv.append_message(conv.roles[1], "Let me think step by step. " + reasoning_part)
    
    conv.append_message(conv.roles[0], "Question: " + target_question)
    conv.append_message(conv.roles[1], None)
    if "NanGPT" in template_name:
        prompted = conv.get_prompt() + " Let me think step by step. "
    return prompted

def preprocess_zero_shot_mmlu(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    input_prompt = question_prompt
    q = questions[-1]
    last_question = q.split("\n\n")[0]
    target_question = q.split("\n\n")[1].replace("A: Let's think step by step.","").strip()
    input_prompt += "\n\nQuestion: " + target_question
    conv.append_message(conv.roles[0], input_prompt.strip())
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt() + " Let me think step by step."
    return prompted

def preprocess_single_turn_mmlu(input_string,template_name = "NanGPT"):
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    input_prompt = question_prompt + " Please solve it step by step carefully."
    for q in questions[:-1]:
        input_part = "\n\nQuestion: " +  q.split("\nA: Let's think step by step. ")[0]
        input_prompt += input_part

        reasoning_part = "\nAnswers: Let me think step by step. " + q.split("\nA: Let's think step by step. ")[1]
        input_prompt +=  reasoning_part

    
    q = questions[-1]
    last_question = q.split("\n\n")[0]
    target_question = q.split("\n\n")[1].replace("A: Let's think step by step.","").strip()

    input_part = "\n\nQuestion: " + last_question.split("\nA: Let's think step by step. ")[0]
    input_prompt +=    input_part
    reasoning_part = "\nAnswers: Let me think step by step. " + last_question.split("\nA: Let's think step by step. ")[1]
    input_prompt += reasoning_part
    
    input_prompt += "\n\nQuestion: " + target_question + "\nAnswers: Let me think step by step. "
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()

    return prompted

def preprocess_few_shot_AO_mmlu(input_string,template_name = "NanGPT"):
    # print(input_string)
    # exit()
    question_prompt = input_string.split("\n\nQ: ")[0]
    questions = input_string.split("\n\nQ: ")[1:]
    conv = get_conv_template(template_name)
    input_prompt = question_prompt
    for q in questions[:-1]:
        input_part = "\n\nQuestion: " +  q.split("\nA: Let's think step by step. ")[0]
        input_prompt += input_part

        reasoning_part = "\nAnswers: " + q.split("\nA: Let's think step by step. ")[1].split("The answer is ")[1]
        input_prompt +=  reasoning_part

    q = questions[-1]
    
    last_question = q.split("\n\n")[0]
    target_question = q.split("\n\n")[1].replace("A: Let's think step by step.","").strip()
    
    input_part = "\n\nQuestion: " + last_question.split("\nA: Let's think step by step.")[0]
    input_prompt +=    input_part
    reasoning_part = "\nAnswers: " + last_question.split("\nA: Let's think step by step. ")[1].split("The answer is ")[1]
    input_prompt += reasoning_part

    input_prompt += "\n\nQuestion: " + target_question + "\nAnswers: " 
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompted = conv.get_prompt()

   
    return prompted


#preprocess_zero_shot_mmlu
def return_preprocess_function(name):
    if name == 'dumpnode':
        return None
    if name == 'raw':
        return preprocess_raw

    if name == 'zero_shot_xcopa':
        return 'zero_shot_xcopa'
    if name == 'few_shot_xcopa':
        return 'few_shot_xcopa'
    if name == 'zero_shot_mmmlu':
        return 'zero_shot_mmmlu'
    if name == 'few_shot_mmmlu':
        return 'few_shot_mmmlu'

    if name == 'zero_shot_bbh_mc':
        return preprocess_zero_shot_bbh_mc
    if name == 'zero_shot_bbh':
        return preprocess_zero_shot_bbh
    if name == 'single_turn_bbh':
        return preprocess_single_turn_bbh
    if name == 'few_shot_AO_bbh':
        return preprocess_few_shot_AO_bbh
    if name == 'few_shot_AO_mmlu':
        return preprocess_few_shot_AO_mmlu
    if name == 'zero_shot_mmlu':
        return preprocess_zero_shot_mmlu
    if name == 'single_turn_mmlu':
        return preprocess_single_turn_mmlu
    if name == 'multi_turn_mmlu':
        return preprocess_multi_turn_mmlu
