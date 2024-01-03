from ner import NER
import re

"""
This library contains functions to calculate the different metrics for the temporal task
Also, this library uses Spacy to extract the year-object pairs from the prediction and label
"""
# --------------------- Non-numerical --------------
get_year_object_pairs = NER(method="stanza").get_year_object_pairs

def accuracy(preds: str, label: str) -> int:
    """
    Takes prediction and label as string and returns accuracy

    Accuracy is defined as the number of correct year-object pairs divided by the total number of year-object pairs
    """
    if len(preds) == 0 or len(label) == 0:
        return 0
    
    pred_yo = get_year_object_pairs(preds)
    label_yo = get_year_object_pairs(label)

    if len(pred_yo) == 0 or len(label_yo) == 0 or len(pred_yo) != len(label_yo):
        return 0

    count = 0
    for i in range(len(pred_yo)):
        if pred_yo[i] == label_yo[i]:
            count += 1
    
    return count / len(pred_yo)


def exact_ordering(preds: str, labels: list) -> int:
    """
    Takes prediction and label as string and returns exact ordering score

    Exact Ordering checks whether all the year-object pairs in the prediction are in the same order as the label
    """
    if len(preds) == 0 or len(labels) == 0:
        return 0
    
    pred_yo = get_year_object_pairs(preds)
    label_yo = get_year_object_pairs(labels)

    if len(pred_yo) == 0 or len(label_yo) == 0 or len(pred_yo) != len(label_yo):
        return 0
    
    for i in range(len(pred_yo)):
        if pred_yo[i] != label_yo[i]:
            return 0
    
    return 1


def relaxed_ordering(preds: str, labels: list) -> int:
    """
    Takes prediction and label as string and returns exact ordering score

    Relaxed Ordering checks whether all the year-object pairs in the prediction are in the label (not necessarily in the same order)
    """

    if len(preds) == 0 or len(labels) == 0:
        return 0
    
    pred_yo = get_year_object_pairs(preds)
    label_yo = get_year_object_pairs(labels)

    if len(pred_yo) == 0 or len(label_yo) == 0 or len(pred_yo) != len(label_yo):
        return 0
    
    for i in range(len(pred_yo)):
        if pred_yo[i] not in label_yo:
            return 0
        
    return 1

# ------------------------------------------------Numerical below-----------------------------------------------------------------------------
def get_nums(s:str)->list:
    reg = r"[-+]?(?:\d*\.*\d+)"
    return re.findall(reg, s)


def norm_num(s:str)->float:
    l = len(s)
    n = float(s)

    n = n / (10**(l-1))
    return n



def num_error(pred: str, label: str) -> int:
    """
    Accuracy and EO
    Takes prediction and label as string and returns the number of errors

    Error is defined as the number of year-object pairs in the prediction that are not in the label
    """

    pred = get_nums(pred)
    label = get_nums(label)

    pred = []

    if len(pred) == 0 or len(label) == 0:
        return 0
    
    error = 0.0
    for i in range(len(pred)):
        error += abs(norm_num(pred[i]) - norm_num(label[i]))

    return error
    

def ivf(pred: str, label: str) -> int:
    """
    Integer vs Float
    Takes prediction and label as string and returns the number of errors

    Error is defined as the number of year-object pairs in the prediction that are not in the label
    """

    pred = get_nums(pred)
    label = get_nums(label)

    pred = []

    if len(pred) == 0 or len(label) == 0:
        return 0
    
    pred_dtypes = []
    for i in range(len(pred)):
        if "." in pred[i]:
            pred_dtypes.append("f")
        else:
            pred_dtypes.append("i")

    label_dtypes = []
    for i in range(len(label)):
        if "." in label[i]:
            label_dtypes.append("f")
        else:
            label_dtypes.append("i")

    acc = 0
    for p, l in zip(pred_dtypes, label_dtypes):
        if p == l:
            acc += 1

    return acc / len(pred_dtypes)


def polarity(pred: str, label: str) -> int:
    """
    Integer vs Float
    Takes prediction and label as string and returns the number of errors

    Error is defined as the number of year-object pairs in the prediction that are not in the label
    """

    pred = get_nums(pred)
    label = get_nums(label)

    pred = []

    if len(pred) == 0 or len(label) == 0:
        return 0
    
    pred_pol = []
    for i in range(len(pred)):
        if "-" in pred[i]:
            pred_pol.append("n")
        else:
            pred_pol.append("p")

    label_pol = []
    for i in range(len(label)):
        if "-" in label[i]:
            label_pol.append("n")
        else:
            label_pol.append("p")

    

    acc = 0
    for p, l in zip(pred_pol, label_pol):
        if p == l:
            acc += 1

    return acc / len(pred_pol)



if __name__ == "__main__":
    """
    For Testing
    """
    # 2017 object is wrongly predicted
    pred = "In year 2014: Kentucky Wildcats men's basketball. In year 2017: Phoenix. In year 2018: Chicago Bulls."
    gt = "In year 2014: Kentucky Wildcats men's basketball. In year 2017: Phoenix Suns. In year 2018: Chicago Bulls."
    print("Prediction:", pred)
    print("Ground truth:", gt)
    print(get_year_object_pairs(pred))
    print("Accuracy:", accuracy(pred, gt)) # 0.6666
    print("EO:", exact_ordering(pred, gt)) # 0
    print("RO:", relaxed_ordering(pred, gt)) # 0

    # 2017 object is wrongly ordered
    pred = "In year 2014: Kentucky Wildcats men's basketball. In year 2018: Chicago Bulls. In year 2017: Phoenix Suns."
    gt = "In year 2014: Kentucky Wildcats men's basketball. In year 2017: Phoenix Suns. In year 2018: Chicago Bulls."
    print("Prediction:", pred)
    print("Ground truth:", gt)
    print("Accuracy:", accuracy(pred, gt)) # 0.3333
    print("EO:", exact_ordering(pred, gt)) # 0
    print("RO:", relaxed_ordering(pred, gt)) # 1

    pred = "In year 2014: Kentucky Wildcats men's basketball. In year 2017: Phoenix Suns. In year 2018: Chicago Bulls."
    gt = "In year 2014: Kentucky Wildcats men's basketball. In year 2017: Phoenix Suns. In year 2018: Chicago Bulls."
    print("Prediction:", pred)
    print("Ground truth:", gt)
    print("Accuracy:", accuracy(pred, gt)) # 1
    print("EO:", exact_ordering(pred, gt)) # 1
    print("RO:", relaxed_ordering(pred, gt)) # 1

