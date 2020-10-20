import sys
from unsup_spatial_pred import Evaluator


def run(path):
    evaluator = Evaluator(path)
    evaluator.evaluate()
    evaluator.display_results()


if __name__ == '__main__':
    run(sys.argv[1])
