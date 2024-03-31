import torch

from verify_common import print_q15, nchw2nhwc

def main():
    torch.manual_seed(0)
    torch.set_printoptions(threshold=10_000)

    X = torch.rand(1, 4, 4, 4) / 2
    Y = torch.rand(1, 4, 4, 4) / 2

    outputs = torch.add(X, Y)

    print('X')
    print_q15(X)
    print('X NHWC')
    print_q15(nchw2nhwc(X))
    print('Y')
    print_q15(Y)
    print('Y NHWC')
    print_q15(nchw2nhwc(Y))
    print('outputs')
    print_q15(outputs)

if __name__ == '__main__':
    main()

