import glob
import re

def main():
    with open('cost_models.txt', 'w', newline='\n') as output:
        for filename in sorted(glob.glob('cnn/*.c')):
            output.write(f'{filename}\n')
            with open(filename) as f:
                in_cost_model = False
                for line in f:
                    if re.search('BEGIN.*COST MODEL', line):
                        in_cost_model = True
                    if in_cost_model:
                        output.write(line)
                    if re.search('END.*COST MODEL', line):
                        in_cost_model = False
                        output.write('\n')

                output.write('\n')

if __name__ == '__main__':
    main()
