import csv
from os.path import dirname, realpath
import sys

from matplotlib import pyplot as plt

sys.path.append(dirname(dirname(realpath(__file__))))

from settings import Settings, arg_parser

def main():
    global_settings = Settings()
    global_settings = arg_parser(global_settings)

    logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + global_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + "_evo_search.csv"

    logfile = open(logfname, 'r')
    reader = csv.reader(logfile)

    x = []
    y1 = []
    y2 = []

    max_iter = 0
    for row_idx, row in enumerate(reader):
        time, iter, best_score, worst_score, best_acc, worst_acc, best_imc, worst_imc, best_config, worst_config = row
        # Skip the header
        if row_idx == 0:
            continue

        iter = int(iter)
        best_score = float(best_score)
        worst_score = float(worst_score)

        x.append(iter)
        y1.append(best_score)
        y2.append(worst_score)

        max_iter = iter

    handles = []
    handles.append(plt.plot(x, y1, label='Best score in population'))
    handles.append(plt.plot(x, y2, label='Worst score in population'))
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.xticks(range(0, max_iter, max_iter//10))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
