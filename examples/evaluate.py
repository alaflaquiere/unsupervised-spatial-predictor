import sys
from unsup_spatial_pred import Evaluator


def run(path):
    evaluator = Evaluator(path)
    evaluator.plot_experiment_stats()


if __name__ == '__main__':
    run(sys.argv[1])
