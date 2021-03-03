import os


def main_evaluate():
    code = os.system('bash evaluate_trained_model.sh')
    assert code == 0


def main_vocab():
    code = os.system('bash create_vocabs.sh')
    assert code == 0


if __name__ == '__main__':
    main_evaluate()
