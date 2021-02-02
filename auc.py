from sklearn.metrics import roc_auc_score

def measure_auc(label, pred, num_classes):
    aucss = []
    for i in list(range(num_classes)):

        if (label[:, i] == 0).all()or(label[:, i] == 1).all():
            aucss.append('--')
        else:
            aucss.append(roc_auc_score(label[:, i], pred[:, i]))
    return aucss

def presision(label, pred,num_classes,threshold):
    pre=[]
    for i in range(num_classes):
        tp, fp, fn, tn=get_confusion_matrix(label[:,i], pred[:,i],threshold)
        pre.append(tp /( tp + fp+0.0000001))
    return pre

def recall(label, pred,num_classes,threshold):
    rec=[]
    for i in range(num_classes):
        tp, fp, fn, tn=get_confusion_matrix(label[:,i], pred[:,i],threshold)
        rec.append(tp / (tp + fn+0.0000001))
    return rec

def accuracy(label, pred,num_classes,threshold):
    acc=[]
    for i in range(num_classes):
        tp, fp, fn, tn=get_confusion_matrix(label[:,i], pred[:,i],threshold)
        acc.append((tp + tn) / (tp + fn + fp + tn))
    return acc

def get_confusion_matrix(labels, preds,threshold):
    """
    计算混淆矩阵
    """

    tp, fp, fn, tn = 0., 0., 0., 0.
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] >= threshold:
            tp += 1
        elif labels[i] == 0 and preds[i] >= threshold:
            fp += 1
        elif labels[i] == 1 and preds[i] < threshold:
            fn += 1
        else:
            tn += 1
    #print(tp, fp, fn, tn)
    return tp, fp, fn, tn

if __name__ == '__main__':
    import numpy as np
    y=np.array([[1],[0],[0],[1],[0]])
    y_p=np.array([[0.6],[0.85],[0.1],[0.8],[0.4]])
    auc=measure_auc(y, y_p, 1)
    print(auc)
