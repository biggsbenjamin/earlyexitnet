import random
import os
import argparse
from main import train_n_test

def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

def main():
    parser = argparse.ArgumentParser(description="Early Exit CLI")

    parser.add_argument('-m','--model_name',
            choices=[   'b_lenet',
                        'b_lenet_fcn',
                        'b_lenet_se',
                        'lenet',
                        'testnet',
                        'brnfirst',
                        'brnfirst_se',
                        'brnsecond',
                        'backbone_se',
                        'b_lenet_cifar',
                        ],
            required=True, help='select the model name')

    parser.add_argument('-mp','--trained_model_path',metavar='PATH',type=path_check,
            required=False,
            help='Path to previously trained model to load, the same type as model name')

    parser.add_argument('-bbe','--bb_epochs', metavar='N',type=int, default=1, required=False,
            help='Epochs to train backbone separately, or non ee network')
    parser.add_argument('-jte','--jt_epochs', metavar='N',type=int, default=1, required=False,
            help='Epochs to train exits jointly with backbone')
    parser.add_argument('-rn', '--run_notes', type=str, required=False,
            help='Some notes to add to the train/test information about the model or otherwise')

    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for training, NOT CURRENTLY USED')

    parser.add_argument('-d','--dataset',
            choices=['mnist','cifar10','cifar100'],
            required=False, default='mnist',
            help='select the dataset, default is mnist')
    #threshold inputs for testing
    parser.add_argument('-t1','--top1_threshold',type=float,required=False)
    parser.add_argument('-entr','--entr_threshold',type=float,required=False)

    #TODO arguments to add
        #batch size training
        #batch size testing
        #training loss function
        #some kind of testing specification

    # parse the arguments
    args = parser.parse_args()

    train_n_test(args)

if __name__ == "__main__":
    main()
