import torch
import torch.nn as nn

from verify_common import print_q15, nchw2nhwc

def main():
    torch.manual_seed(0)
    torch.set_printoptions(threshold=10_000)

    inputs = torch.rand(1, 4, 32, 32) - 0.5

    outputs = nn.functional.relu(inputs)

    print('inputs')
    print_q15(inputs)
    print('inputs NHWC')
    print_q15(nchw2nhwc(inputs))
    print('outputs')
    print_q15(outputs)

if __name__ == '__main__':
    main()
