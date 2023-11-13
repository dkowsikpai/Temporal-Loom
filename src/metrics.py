from ner import NER

"""
This library contains functions to calculate the different metrics for the temporal task
Also, this library uses Spacy to extract the year-object pairs from the prediction and label
"""

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

