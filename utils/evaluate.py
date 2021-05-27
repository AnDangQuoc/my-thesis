import numpy as np
import seg_metrics.seg_metrics as sg

LABELS = [1, 2, 3]


def eval(groundTruth, predict, outFile='result.csv'):

    metrics = sg.write_metrics(labels=LABELS,
                               gdth_path=groundTruth,
                               pred_path=predict, csv_file=outFile)
    return metrics
