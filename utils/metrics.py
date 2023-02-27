import numpy as np


class RunningMetrics_CD(object):
    def __init__(self, num_classes=2):
        """
        Computes and stores the Metric values from Confusion Matrix
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param num_classes: <int> number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def __fast_hist(self, label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < self.num_classes)
        hist = np.bincount(self.num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_gts, label_preds):
        """
        Compute Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gts: <np.ndarray> ground-truths
        :param label_preds: <np.ndarray> predictions
        :return:
        """
        for lt, lp in zip(label_gts, label_preds):
            self.confusion_matrix += self.__fast_hist(lt.flatten(), lp.flatten())

    def reset(self):
        """
        Reset Confusion Matrix
        :return:
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_cm(self):
        return self.confusion_matrix

    def get_scores(self):
        """
        Returns score about:
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :return:
        """
        hist = self.confusion_matrix
        tp = hist[1, 1]
        fn = hist[1, 0]
        fp = hist[0, 1]
        tn = hist[0, 0]
        # acc
        oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
        # recall
        recall = tp / (tp + fn + np.finfo(np.float32).eps)
        # precision
        precision = tp / (tp + fp + np.finfo(np.float32).eps)
        # F1 score
        f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
        # IoU
        iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
        # pre
        pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
        # kappa
        kappa = (oa - pre) / (1 - pre)
        score_dict = {'precision': precision, 'recall': recall, 'OA': oa, 'IoU': iou, 'Kappa': kappa, 'F1': f1}
        # scores = {'Overall_Acc': acc,
        #           'Mean_IoU': mean_iu}
        # # 'Kappa': kappa}
        # scores.update(cls_iu)
        # scores.update({'precision_1': precision[1],
        #                'recall_1': acc_cls_[1],
        #                'F1_1': F1[1]})
        return score_dict


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
