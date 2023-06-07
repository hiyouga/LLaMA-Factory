def prompt_template_alpaca(query, history=None):
    prompt = ""
    if history:
        for old_query, response in history:
            prompt += "Human:{}\nAssistant:{}\n".format(old_query, response)
    prompt += "Human:{}\nAssistant:".format(query)
    return prompt


def prompt_template_ziya(query, history=None):
    prompt = ""
    if history:
        for old_query, response in history:
            prompt += "<human>:{}\n<bot>:{}\n".format(old_query, response)
    prompt += "<human>:{}\n<bot>:".format(query)
    return prompt
