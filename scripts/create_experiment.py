"""
新規実験 作成 スクリプト
"""
from pathlib import Path
import argparse

# dir関連
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent

def main():
    """
    entry point
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n',
        type=str, help='name of experiment.', required=True)
    parser.add_argument('--type', '-t', choices=('chainer', 'none'),
        type=str, help='type of experiment.', required=True)
    args = parser.parse_args()

    print('# Create New Experiment')
    print('Name: {}'.format(args.name))
    print('Type: {}'.format(args.type))

    ex_path = ROOT_PATH.joinpath('experiments', args.name)
    result_path = ROOT_PATH.joinpath('results', args.name)
    # make directory
    ex_path.mkdir(exist_ok=True)
    result_path.mkdir(exist_ok=True)
    result_path.joinpath('.gitkeep').touch()
    print('create:', ROOT_PATH.joinpath('experiments', args.name))
    print('create:', ROOT_PATH.joinpath('results', args.name))

    if args.type == 'chainer':
        print('# add chainer file')
        ex_path.joinpath('train_{}.py'.format(args.name)).touch()
        ex_path.joinpath('dataset.py').touch()
        ex_path.joinpath('net.py').touch()
        ex_path.joinpath('updater.py').touch()
        ex_path.joinpath('visualize.py').touch()
        print('create, train_{}.py, dataset.py, net.py, updater.py, visualize.py'.format(args.name))
        # todo
        # 将来的にはファイルの中身を書く, 現在は忙しいのでやんないよ

if __name__ == "__main__":
    main()
