import sys
from unsup_spatial_pred import Evaluator


def run(path):
    evaluator = Evaluator(path)
    evaluator.plot_experiment_stats()
    evaluator.plot_embedding(0)
    evaluator.plot_pairwise_distances_comparison(0)


if __name__ == '__main__':
    run(sys.argv[1])
