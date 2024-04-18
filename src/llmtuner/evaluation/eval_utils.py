
def fs_cothub_bbh_match_answer(task_data, response):
    # CoT hub match answer for BBH
    # https://github.com/FranxYao/chain-of-thought-hub/blob/main/BBH/run_bbh_gpt_3.5_turbo.py

    ans_line = response.split('answer is ')

    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return False, response
    else:
        ans = ans_line[-1].strip()

    if task_data["options"]:
        # Multiple choice, find appearing letter
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']

        for option in options:
            if option in ans:
                return True, option

        return False, ans
    else:
        # Free form, direct return
        if len(ans) and ans[-1] == '.':
            ans = ans[:-1]

        return True, ans



def fs_cothub_mmlu_match_answer(task_data, response):
    ans_line = response.split('answer is')

    # Expect to see 'answer is'. If not return C
    if len(ans_line) == 1:
        return False, "(C)"
    else:
        ans = ans_line[-1].strip()
        
    options = ['(A)', '(B)', '(C)', '(D)']
    for option in options:
        if option in ans:
            return True, option
    options = ['a','b','c','d']
    for i in ans.lower():
        if i in options:
            return True, "({})".format(i.upper())
    
    return False, "(C)"
    