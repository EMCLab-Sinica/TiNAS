import torch
import torch.nn as nn

from verify_common import print_q15, nchw2nhwc

def main():
    torch.manual_seed(0)
    torch.set_printoptions(threshold=10_000)

    inputs = torch.cat(
        [
            torch.maximum(torch.rand(1, 2, 4, 4) - 0.2, torch.Tensor([    -1])),
            torch.minimum(torch.rand(1, 2, 4, 4) + 0.2, torch.Tensor([ 0.999])),
        ], dim=1
    )

    pool = nn.AdaptiveAvgPool2d(1)

    outputs = pool(inputs)

    print('inputs')
    print_q15(inputs)
    print('inputs NHWC')
    print_q15(nchw2nhwc(inputs))
    print('outputs')
    print_q15(outputs)

if __name__ == '__main__':
    main()
