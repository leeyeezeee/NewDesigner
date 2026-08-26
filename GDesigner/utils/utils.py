import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    pred = preds[1] if answer_flag else preds[-1]

    pred = pred.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", pred)
    if not pred:
        return INVALID_ANS

    pred = pred[0] if answer_flag else pred[-1]
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred
