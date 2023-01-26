#training, testing for branchynet-pytorch version

from models.Branchynet import ConvPoolAc,B_Lenet,B_Lenet_fcn,B_Lenet_se, B_Lenet_cifar,B_Alexnet_cifar
from models.Lenet import Lenet
from models.Testnet import Testnet, BrnFirstExit, BrnSecondExit, Backbone_se

from tools import MNISTDataColl, CIFAR10DataColl, CIFAR100DataColl
from tools import Tracker, LossTracker, AccuTracker
from tools import save_model, load_model, shape_test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt
import time

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

#TODO might merge exit+backbone for code reuse
def train_backbone(model, train_dl, valid_dl, batch_size, save_path, epochs=50,
                    loss_f=nn.CrossEntropyLoss(), opt=None, dat_norm=False,
                    device=None):
    #train network backbone

    if opt is None:
        #set to branchynet default
        #Adam algo - step size alpha=0.001
        lr = 0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]
        backbone_params = [
                {'params': model.backbone.parameters()},
                {'params': model.exits[-1].parameters()}
                ]

        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)

    best_val_loss = [1.0, '']
    best_val_accu = [0.0, '']
    trainloss_trk = LossTracker(batch_size,1)
    trainaccu_trk = AccuTracker(batch_size,1)
    validloss_trk = LossTracker(batch_size,1)
    validaccu_trk = AccuTracker(batch_size,1)

    if device is None:
        device = torch.device("cpu")

    # set device model - should be cpu as default
    model.to(device)

    for epoch in range(epochs):
        model.train()
        print("Starting epoch:", epoch+1, end="... ", flush=True)
        train_loss=0.0
        correct_count=0

        trainloss_trk.reset_tracker()
        trainaccu_trk.reset_tracker()
        validloss_trk.reset_tracker()
        validaccu_trk.reset_tracker()

        #training loop
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            results = model(xb)
            #loss for backbone ignores other exits
            #Wasting some forward compute of early exits
            #but shouldn't be included in backward step
            #since params not looked at by optimiser
            #TODO add backbone only method to bn class
            loss = loss_f(results[-1], yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            trainloss_trk.add_loss(loss.item())
            trainaccu_trk.update_correct(results[-1],yb)

        tr_loss_avg = trainloss_trk.get_avg(return_list=True)[-1]
        t1acc = trainaccu_trk.get_avg(return_list=True)[-1]

        #validation
        model.eval()
        with torch.no_grad():
            for xb,yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                res_v = model(xb)
                validloss_trk.add_loss(loss_f(res_v[-1], yb))
                validaccu_trk.update_correct(res_v[-1],yb)

        val_loss_avg = validloss_trk.get_avg(return_list=True)[-1]#should be last of 1
        val_accu_avg = validaccu_trk.get_avg(return_list=True)[-1]

        print(  "T Loss:",tr_loss_avg,
                "T T1 Acc: ", t1acc,
                "V Loss:", val_loss_avg,
                "V T1 Acc:", val_accu_avg)
        if dat_norm:
            file_prefix = "dat_norm-backbone-"
        else:
            file_prefix = "backbone-"
        savepoint = save_model(model, save_path, file_prefix=file_prefix+str(epoch+1), opt=opt,
                tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

        if val_loss_avg < best_val_loss[0]:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
        if val_accu_avg > best_val_accu[0]:
            best_val_accu[0] = val_accu_avg
            best_val_accu[1] = savepoint
    print("BEST VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    print("BEST VAL ACCU: ", best_val_accu[0], " for epoch: ", best_val_accu[1])

    #return best_val_loss[1], savepoint #link to best val loss model
    # Saving best path in text file

    return best_val_accu[1], savepoint #link to best val accu model - trying for now
def train_exits(model, epochs=100):
    #train the exits alone

    #Adam algo - step size alpha=0.001
    #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
    return #something trained

def train_joint(model, train_dl, valid_dl, batch_size, save_path, opt=None,
                loss_f=nn.CrossEntropyLoss(), backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True, dat_norm=False,
                device=None):

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

    if pretrain_backbone:
        if device is None:
            device = torch.device("cpu")
        print("PRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        # forcing opt to be none so only bb params are trained
        best_bb_path,_ = train_backbone(model, train_dl,
                valid_dl, batch_size, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f,dat_norm=dat_norm,
                opt=None,device=device)
        #train the rest...
        print("LOADING BEST BACKBONE:",best_bb_path)
        load_model(model, best_bb_path)
        print("JOINT TRAINING WITH PRETRAINED BACKBONE")
        prefix = 'pretrn-joint'
        bb_path=best_bb_path
    else:
        #jointly trains backbone and exits from scratch
        print("JOINT TRAINING FROM SCRATCH")
        folder_path = 'jnt_fr_scrcth' + timestamp
        prefix = 'joint'
        bb_path="no bb pre-training"

    spth = os.path.join(save_path, folder_path)

    #set up the joint optimiser
    if opt is None: #TODO separate optim function to reduce code, maybe pass params?
        #set to branchynet default
        lr = 0.001 #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]

        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    best_val_loss = [[1.0,1.0], ''] #TODO make sure list size matches num of exits
    best_val_accu = [[0.0,0.0], ''] #TODO make sure list size matches num of exits
    train_loss_trk = LossTracker(train_dl.batch_size,bins=2)
    train_accu_trk = AccuTracker(train_dl.batch_size,bins=2)
    valid_loss_trk = LossTracker(valid_dl.batch_size,bins=2)
    valid_accu_trk = AccuTracker(valid_dl.batch_size,bins=2)

    # set device model - should be cpu as default
    if device is None:
        device = torch.device("cpu")
    model.to(device)

    for epoch in range(joint_epochs):
        model.train()
        print("starting epoch:", epoch+1, end="... ", flush=True)
        train_loss = [0.0,0.0]
        correct_count = [0,0]
        train_loss_trk.reset_tracker()
        train_accu_trk.reset_tracker()
        #training loop
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            results = model(xb)

            raw_losses = [loss_f(res,yb) for res in results]

            losses = [weighting * raw_loss
                        for weighting, raw_loss in zip(model.exit_loss_weights,raw_losses)]

            opt.zero_grad()
            #backward
            for loss in losses[:-1]: #ee losses need to keep graph
                loss.backward(retain_graph=True)
            losses[-1].backward() #final loss, graph not required
            opt.step()

            for i,_ in enumerate(train_loss):
                #weighted losses
                train_loss[i]+=losses[i].item()
            #raw losses
            train_loss_trk.add_loss([exit_loss.item() for exit_loss in raw_losses])
            train_accu_trk.update_correct(results,yb)


        tr_loss_avg_weighted = [loss/(len(train_dl)*batch_size) for loss in train_loss]
        tr_loss_avg = train_loss_trk.get_avg(return_list=True)
        t1acc = train_accu_trk.get_accu(return_list=True)

        #validation
        model.eval()
        with torch.no_grad():
            for xb,yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                res = model(xb)
                valid_loss_trk.add_loss([loss_f(exit, yb) for exit in res])
                valid_accu_trk.update_correct(res,yb)

        val_loss_avg = valid_loss_trk.get_avg(return_list=True)
        val_accu_avg = valid_accu_trk.get_accu(return_list=True)

        print("raw t loss:{} t1acc:{}\nraw v loss:{} v accu:{}".format(
            tr_loss_avg,t1acc,val_loss_avg,val_accu_avg))
        if dat_norm:
            prefix = "dat_norm-"+prefix
        savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt,
            tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

        el_total=0.0
        bl_total=0.0
        for exit_loss, best_loss,l_w in zip(val_loss_avg,best_val_loss[0],model.exit_loss_weights):
            el_total+=exit_loss*l_w
            bl_total+=best_loss*l_w
        #selecting "best" network
        if el_total < bl_total:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint

        ea_total=0.0
        ba_total=0.0
        for exit_accu, best_accu,l_w in zip(val_accu_avg,best_val_accu[0],model.exit_loss_weights):
            ea_total+=exit_accu*l_w
            ba_total+=best_accu*l_w
        #selecting "best" network
        if ea_total > ba_total:
            best_val_accu[0] = val_accu_avg
            best_val_accu[1] = savepoint

    print("BEST* VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    print("BEST* VAL ACCU: ", best_val_accu[0], " for epoch: ", best_val_accu[1])
    #return best val loss path link
    #return best_val_loss[1],savepoint
    return best_val_accu[1],savepoint,bb_path

class Tester:
    def __init__(self,model,test_dl,loss_f=nn.CrossEntropyLoss(),exits=2,
            top1acc_thresholds=[],entropy_thresholds=[],device=None):
        self.model=model
        self.test_dl=test_dl
        self.loss_f=loss_f
        self.exits=exits
        self.sample_total = len(test_dl)
        self.top1acc_thresholds = top1acc_thresholds
        self.entropy_thresholds = entropy_thresholds
        if device is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = device


        if exits > 1:
            #TODO make thresholds a more flexible param
            #setting top1acc threshold for exiting (final exit set to 0)
            #self.top1acc_thresholds = [0.995,0]
            #setting entropy threshold for exiting (final exit set to LARGE)
            #self.entropy_thresholds = [0.025,1000000]
            #set up stat trackers
            #samples exited
            self.exit_track_top1 = Tracker(test_dl.batch_size,exits,self.sample_total)
            self.exit_track_entr = Tracker(test_dl.batch_size,exits,self.sample_total)
            #individual accuracy over samples exited
            self.accu_track_top1 = AccuTracker(test_dl.batch_size,exits)
            self.accu_track_entr = AccuTracker(test_dl.batch_size,exits)

        #total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(test_dl.batch_size,exits,self.sample_total)

        self.top1_pc = None # % exit for top1 confidence
        self.entr_pc = None # % exit for entropy confidence
        self.top1_accu = None #accuracy of exit over exited samples
        self.entr_accu = None #accuracy of exit over exited samples
        self.full_exit_accu = None #accuracy of the exits over all samples
        self.top1_accu_tot = None #total accuracy of network given exit strat
        self.entr_accu_tot = None #total accuracy of network given exit strat

    def _test_multi_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                self.accu_track_totl.update_correct(res,yb)
                for i,(exit,thr) in enumerate(zip(res,self.top1acc_thresholds)):
                    softmax = nn.functional.softmax(exit,dim=-1)
                    sftmx_max = torch.max(softmax)
                    if sftmx_max > thr:
                        #print("top1 exited at exit {}".format(i))
                        self.exit_track_top1.add_val(1,i)
                        self.accu_track_top1.update_correct(exit,yb,bin_index=i)
                        break
                for i,(exit,thr) in enumerate(zip(res,self.entropy_thresholds)):
                    softmax = nn.functional.softmax(exit,dim=-1)
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    if entr < thr:
                        #print("entr exited at exit {}".format(i))
                        self.exit_track_entr.add_val(1,i)
                        self.accu_track_entr.update_correct(exit,yb,bin_index=i)
                        break
    def _test_single_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                self.accu_track_totl.update_correct(res,yb)
    def debug_values(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                for i,exit in enumerate(res):
                    #print("raw exit {}: {}".format(i, exit))
                    softmax = nn.functional.softmax(exit,dim=-1)
                    #print("softmax exit {}: {}".format(i, softmax))
                    sftmx_max = torch.max(softmax)
                    print("exit {} max softmax: {}".format(i, sftmx_max))
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    print("exit {} entropy: {}".format(i, entr))
                    #print("exit CE loss: {}".format(loss_f(exit,yb)))

    def test(self):
        print(f"Test of length {self.sample_total} starting")
        if self.exits > 1:
            self._test_multi_exit()
            self.top1_pc = self.exit_track_top1.get_avg(return_list=True)
            self.entr_pc = self.exit_track_entr.get_avg(return_list=True)
            self.top1_accu = self.accu_track_top1.get_accu(return_list=True)
            self.entr_accu = self.accu_track_entr.get_accu(return_list=True)
            self.top1_accu_tot = np.sum(self.accu_track_top1.val_bins)/self.sample_total
            self.entr_accu_tot = np.sum(self.accu_track_entr.val_bins)/self.sample_total
        else:
            self._test_single_exit()
        #accuracy of each exit over FULL data set
        self.full_exit_accu = self.accu_track_totl.get_accu(return_list=True)
        #TODO save test stats along with link to saved model

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
    elif args.model_name == 'b_alexnet_cifar':
        model = B_Alexnet_cifar()
        exits = 2
        print(shape_test(model, [3,32,32], [1])) #output is not one hot encoded
    else:
        raise NameError("Model not supported, check name:",args.model_name)
    print("Model done:", args.model_name)

    # Device setup
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    print("Number of workers:",args.num_workers)

    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set")
    batch_size_test = 1 #test bs in branchynet
    normalise=False     #normalise the training data or not
    bb_path=None    # init for notes of pre trained bb

    if args.trained_model_path is not None:
        # NOTE only performing testing on trained model
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
        num_workers = 1 if args.num_workers is None else args.num_workers
        #sort into training, and test data
        if args.dataset == 'mnist':
            datacoll = MNISTDataColl(batch_size_train=batch_size_train,
                    batch_size_test=batch_size_test,normalise=normalise,
                    v_split=validation_split, num_workers=num_workers)
        elif args.dataset == 'cifar10':
            datacoll = CIFAR10DataColl(batch_size_train=batch_size_train,
                    batch_size_test=batch_size_test,normalise=normalise,
                    v_split=validation_split, num_workers=num_workers)
        else:
            raise NameError("Dataset not supported, check name:",args.dataset)
        train_dl = datacoll.get_train_dl()
        valid_dl = datacoll.get_valid_dl()
        print("Got training data, batch size:",batch_size_train)

        #start training loop for epochs - at some point add recording points here
        path_str = 'outputs/'
        print("backbone epochs: {} joint epochs: {}".format(args.bb_epochs, args.jt_epochs))


        pretrain_backbone=True
        if args.bb_epochs == 0:
            # FIXME check if backbone provided (training exits/joint) or joint from scratch
            pretrain_backbone=False

        if exits > 1:
            save_path,last_path,bb_path = train_joint(model, train_dl, valid_dl, batch_size_train,
                    path_str,backbone_epochs=args.bb_epochs,joint_epochs=args.jt_epochs,
                    loss_f=loss_f,pretrain_backbone=pretrain_backbone,dat_norm=normalise,
                    device=device)
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
                    loss_f=loss_f, opt=opt, dat_norm=normalise,
                    device=device)

        #save some notes about the run
        notes_path = os.path.join(os.path.split(save_path)[0],'notes.txt')
        with open(notes_path, 'w') as notes:
            notes.write("bb epochs {}, jt epochs {}\n".format(args.bb_epochs, args.jt_epochs))
            notes.write("Training batch size {}, Test batchsize {}\n".format(batch_size_train,
                                                                           batch_size_test))
            if hasattr(model,'exit_loss_weights'):
                notes.write("model training exit weights:"+str(model.exit_loss_weights)+"\n")
            notes.write("Path to BB model:\n"+str(bb_path)+"\n")
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
    # NOTE back to cpu for testing
    model.to("cpu")

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

    #be nice to have comparison against pytorch pretrained LeNet from pytorch

if __name__ == "__main__":
    train_n_test()
