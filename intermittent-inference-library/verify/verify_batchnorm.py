import torch
import torch.nn as nn

from verify_common import print_q15, nchw2nhwc

def main():
    torch.manual_seed(0)
    torch.set_printoptions(threshold=10_000)

    inputs = torch.rand(1, 4, 32, 32)
    weight = torch.rand(4)
    bias = torch.rand(4)
    mean = torch.rand(4)
    var = torch.rand(4) * 1024
    epsilon = 0.00001
    var_pre_calculated = 1/torch.sqrt(var + epsilon)

    outputs = nn.functional.batch_norm(inputs, mean, var, weight, bias)

    print('inputs')
    print_q15(inputs)
    print('inputs NHWC')
    print_q15(nchw2nhwc(inputs))
    print('weight')
    print_q15(weight)
    print('bias')
    print_q15(bias)
    print('mean')
    print_q15(mean)
    print('var_pre_calculated')
    print_q15(var_pre_calculated)
    print('outputs')
    print_q15(outputs)

if __name__ == '__main__':
    main()
