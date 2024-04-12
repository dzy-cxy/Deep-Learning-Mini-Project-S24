import os
import csv
import data
import models
import options


def test(opt):
    print(opt)
    dataloader = data.get_dataloader(False, opt.batch, opt.dataset_dir)
    model = models.ResNetModel(opt, train=False)
    model.load_model(opt.params_path)

    total_n = 0
    total_correct = 0

    for batch in dataloader:
        inputs, labels = batch
        correct, total, _ = model.test(inputs, labels)
        total_correct += correct
        total_n += total
    
    acc = 100 * total_correct / total_n
    err = 100 - acc
    print(f'accuracy: {acc:.2f} %')
    print(f'error: {err:.2f} %')
    print(f'{total_correct} / {total_n}')


if __name__ == '__main__':
    args = options.parse_args_test()
    test(args)
