"""
CLI for training and testing early-exit and normal CNNs.
"""

import random
import os
import argparse

# import training functions
from earlyexitnet.training_tools.train import train_backbone, train_joint, train_exits

# import testing class and functions
from earlyexitnet.testing_tools.test import Tester

"""
Full function for training and testing, using args in main()
"""
def train_n_test(args):
    #shape testing
    #print(shape_test(model, [1,28,28], [1])) #output is not one hot encoded

    exits = 1 # set number of exits
    #set up the model specified in args
    if args.model_name == 'lenet':
        model = Lenet()
    elif args.model_name == 'testnet':
        model = Testnet()
    elif args.model_name == 'brnfirst': #fcn version
        model = BrnFirstExit()
    elif args.model_name == 'brnsecond': #fcn version
        model = BrnSecondExit()
    elif args.model_name == 'brnfirst_se': #se version
        model = BrnFirstExit_se()
    elif args.model_name == 'backbone_se': #se backbone (for baseline)
        model = Backbone_se()
    elif args.model_name == 'b_lenet':
        model = B_Lenet()
        exits = 2
    elif args.model_name == 'b_lenet_fcn':
        model = B_Lenet_fcn()
        exits = 2
    elif args.model_name == 'b_lenet_se':
        model = B_Lenet_se()
        exits = 2
    elif args.model_name == 'b_lenet_cifar':
        model = B_Lenet_cifar()
        exits = 2
        print(shape_test(model, [3,32,32], [1])) #output is not one hot encoded
    else:
        raise NameError("Model not supported, check name:",args.model_name)
    print("Model done:", args.model_name)

    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set")
    batch_size_test = 1 #test bs in branchynet
    normalise=False     #normalise the training data or not

    if args.trained_model_path is not None:
        #load in the model from the path
        load_model(model, args.trained_model_path)
        #skip to testing
        if args.dataset == 'mnist':
            datacoll = MNISTDataColl(batch_size_test=batch_size_test)
        elif args.dataset == 'cifar10':
            datacoll = CIFAR10DataColl(batch_size_test=batch_size_test)
        else:
            raise NameError("Dataset not supported, check name:",args.dataset)
        #TODO make use of profiling split
        notes_path = os.path.join(os.path.split(args.trained_model_path)[0],'notes.txt')
        save_path = args.trained_model_path

    else:
        #get data and load if not already exiting - MNIST for now
        batch_size_train = 500 #training bs in branchynet
        validation_split = 0.2
        #sort into training, and test data
        if args.dataset == 'mnist':
            datacoll = MNISTDataColl(batch_size_train=batch_size_train,
                    batch_size_test=batch_size_test,normalise=normalise,
                    v_split=validation_split)
        elif args.dataset == 'cifar10':
            datacoll = CIFAR10DataColl(batch_size_train=batch_size_train,
                    batch_size_test=batch_size_test,normalise=normalise,
                    v_split=validation_split)
        else:
            raise NameError("Dataset not supported, check name:",args.dataset)
        train_dl = datacoll.get_train_dl()
        valid_dl = datacoll.get_valid_dl()
        print("Got training data, batch size:",batch_size_train)

        #start training loop for epochs - at some point add recording points here
        path_str = 'outputs/'
        print("backbone epochs: {} joint epochs: {}".format(args.bb_epochs, args.jt_epochs))

        if exits > 1:
            save_path,last_path = train_joint(model, train_dl, valid_dl, batch_size_train,
                    path_str,backbone_epochs=args.bb_epochs,joint_epochs=args.jt_epochs,
                    loss_f=loss_f,pretrain_backbone=True,dat_norm=normalise)
        else:
            #provide optimiser for non ee network
            lr = 0.001 #Adam algo - step size alpha=0.001
            #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
            exp_decay_rates = [0.99, 0.999]
            opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

            ts = dt.now().strftime("%Y-%m-%d_%H%M%S")
            path_str = f'outputs/bb_only/time_{ts}'
            print("Saving to:",path_str)
            save_path,last_path = train_backbone(model, train_dl, valid_dl,
                    batch_size=batch_size_train, save_path=path_str, epochs=args.bb_epochs,
                    loss_f=loss_f, opt=opt, dat_norm=normalise)

        #save some notes about the run
        notes_path = os.path.join(os.path.split(save_path)[0],'notes.txt')
        with open(notes_path, 'w') as notes:
            notes.write("bb epochs {}, jt epochs {}\n".format(args.bb_epochs, args.jt_epochs))
            notes.write("Training batch size {}, Test batchsize {}\n".format(batch_size_train,
                                                                           batch_size_test))
            if hasattr(model,'exit_loss_weights'):
                notes.write("model training exit weights:"+str(model.exit_loss_weights)+"\n")
            notes.write("Path to last model:"+str(last_path)+"\n")
        notes.close()

        #TODO sort out graph of training data
        #separate graphs for pre training and joint training

    test_dl = datacoll.get_test_dl()
    #once trained, run it on the test data
    if exits>1:
        top1_thr = [args.top1_threshold, 0]
        entr_thr = [args.entr_threshold, 1000000]
        net_test = Tester(model,test_dl,loss_f,exits,
                top1_thr,entr_thr)
    else:
        net_test = Tester(model,test_dl,loss_f,exits)
    top1_thr = net_test.top1acc_thresholds
    entr_thr = net_test.entropy_thresholds
    net_test.test()
    #get test results
    test_size = net_test.sample_total
    top1_pc = net_test.top1_pc
    entropy_pc = net_test.entr_pc
    top1acc = net_test.top1_accu
    entracc = net_test.entr_accu
    t1_tot_acc = net_test.top1_accu_tot
    ent_tot_acc = net_test.entr_accu_tot
    full_exit_accu = net_test.full_exit_accu
    #get percentage exits and avg accuracies, add some timing etc.
    print("top1 thrs: {},  entropy thrs: {}".format(top1_thr, entr_thr))
    print("top1 exit %s {},  entropy exit %s {}".format(top1_pc, entropy_pc))
    print("Accuracy over exited samples:")
    print("top1 exit acc % {}, entropy exit acc % {}".format(top1acc, entracc))
    print("Accuracy over network:")
    print("top1 acc % {}, entr acc % {}".format(t1_tot_acc,ent_tot_acc))
    print("Accuracy of the individual exits over full set: {}".format(full_exit_accu))

    with open(notes_path, 'a') as notes:
        notes.write("\n#######################################\n")
        notes.write(f"\nTesting results: for {args.model_name}\n")
        notes.write(f"on dataset {args.dataset}\n")
        notes.write("Test sample size: {}\n".format(test_size))
        notes.write("top1 thrs: {},  entropy thrs: {}\n".format(top1_thr, entr_thr))
        notes.write("top1 exit %s {}, entropy exit %s {}\n".format(top1_pc, entropy_pc))
        notes.write("best* model "+save_path+"\n")
        notes.write("Accuracy over exited samples:\n")
        notes.write("top1 exit acc % {}, entropy exit acc % {}\n".format(top1acc, entracc))
        notes.write("Accuracy over EE network:\n")
        notes.write("top1 acc % {}, entr acc % {}\n".format(t1_tot_acc,ent_tot_acc))
        notes.write("Accuracy of the individual exits over full set: {}\n".format(full_exit_accu))

        if args.run_notes is not None:
            notes.write(args.run_notes+"\n")
    notes.close()


def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

"""
Main function that sorts out the CLI args and runs training and testing function.
"""
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
    #be nice to have comparison against pytorch pretrained LeNet from pytorch

if __name__ == "__main__":
    main()
