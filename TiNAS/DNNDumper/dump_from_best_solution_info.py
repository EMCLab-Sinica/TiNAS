import argparse
import json

from dump_cnn_msp430_extNVM import dump_cnn_handler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best-solution-info', required=True, help='path to best_solution_info.json')
    parser.add_argument('--power-type', choices=['intpow', 'contpow'], required=True, help='Power type to run the model - one of intpow or contpow')
    parser.add_argument('--output-path', default='model.h', help='Path to the output model header file')
    args = parser.parse_args()

    with open(args.best_solution_info) as f:
        best_solution_info = json.load(f)[1]

    if 'subnet_latency_info' in best_solution_info:
        best_solution_info = best_solution_info['subnet_latency_info']

    subnet_obj = best_solution_info['subnet_obj']
    # Just use the intermittent design, as the explorer use the same design for both continuous and intermittent power
    network_exec_design = best_solution_info.get('perf_exec_design_intpow') or best_solution_info['perf_exec_design']
    network_exec_design_contpow = best_solution_info.get('perf_exec_design_contpow_fp')

    dump_cnn_handler(subnet_obj, network_exec_design, network_exec_design_contpow, args.power_type,
                     model_name='MODEL_' + args.power_type.upper(), output_path=args.output_path)

if __name__ == '__main__':
    main()

