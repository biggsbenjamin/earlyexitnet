#training, testing for branchynet-pytorch version

from models.Branchynet import B_Lenet, ConvPoolAc
from models.Lenet import Lenet
from models.Testnet import Testnet, BrnFirstExit, BrnSecondExit

from tools import MNISTDataColl, LossTracker, AccuTracker, save_model, load_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

#TODO might merge exit+backbone for code reuse
def train_backbone(model, train_dl, valid_dl, batch_size, save_path, epochs=50,
                    loss_f=nn.CrossEntropyLoss(), opt=None, dat_norm=False):
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

    #probe params to double check only backbone run
    #probe_params(model) #satisfied that setting specific params works right

    best_val_loss = [1.0, '']
    trainloss_trk = LossTracker(batch_size,1)
    trainaccu_trk = AccuTracker(batch_size,1)
    validloss_trk = LossTracker(batch_size,2)
    validaccu_trk = AccuTracker(batch_size,2)

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

            train_loss+=loss.item()
            correct_count+=get_num_correct(results[-1],yb)

            trainloss_trk.add_loss(loss.item())
            trainaccu_trk.update_correct(results[-1],yb)

        tr_loss_avg = train_loss / (len(train_dl)*batch_size)
        t1acc = correct_count / (len(train_dl)*batch_size)

        trk_tr_loss_avg = trainloss_trk.get_avg(return_list=True)[-1]
        trk_t1acc = trainaccu_trk.get_avg(return_list=True)[-1]

        assert tr_loss_avg == trk_tr_loss_avg, f"{tr_loss_avg} {trk_tr_loss_avg}"
        assert t1acc == trk_t1acc, f"{t1acc} {trk_t1acc}"

        #validation
        model.eval()
        with torch.no_grad():
            vloss_ls = [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]
            vaccu_ls = [[get_num_correct(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]

            valid_losses = np.sum(np.array(vloss_ls), axis=0)
            valid_accus = np.sum(np.array(vaccu_ls), axis=0)

            validloss_trk.add_losses(vloss_ls)
            validaccu_trk.update_correct_list(vaccu_ls)

        val_loss_avg = valid_losses[-1] / (len(valid_dl)*batch_size)
        val_accu_avg = valid_accus[-1] / (len(valid_dl)*batch_size)
        print(len(valid_dl)*batch_size)

        trk_val_loss_avg = validloss_trk.get_avg(return_list=True)[-1]
        trk_val_accu_avg = validaccu_trk.get_avg(return_list=True)[-1]

        assert val_loss_avg == trk_val_loss_avg, f"{val_loss_avg} {trk_val_loss_avg}"
        assert val_accu_avg == trk_val_accu_avg, f"{val_accu_avg} {trk_val_accu_avg}"

        print(  "T Loss:",tr_loss_avg,
                "T1 Acc: ", t1acc,
                "V Loss:", val_loss_avg)
        #probe_params(model)
        if dat_norm:
            file_prefix = "dat_norm-backbone-"
        else:
            file_prefix = "backbone-"
        savepoint = save_model(model, save_path, file_prefix=file_prefix+str(epoch+1), opt=opt)

        if val_loss_avg < best_val_loss[0]:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
    print("BEST VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    return best_val_loss[1], savepoint #link to best val loss model

def train_exits(model, epochs=100):
    #train the exits alone

    #Adam algo - step size alpha=0.001
    #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
    return #something trained

def train_joint(model, train_dl, valid_dl, batch_size, save_path, opt=None,
                loss_f=nn.CrossEntropyLoss(), backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True, dat_norm=False):

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

    if pretrain_backbone:
        print("PRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        best_bb_path,_ = train_backbone(model, train_dl,
                valid_dl, batch_size, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f,dat_norm=dat_norm)
        #train the rest...
        print("LOADING BEST BACKBONE:",best_bb_path)
        load_model(model, best_bb_path)
        print("JOINT TRAINING WITH PRETRAINED BACKBONE")

        prefix = 'pretrn-joint'
    else:
        #jointly trains backbone and exits from scratch
        print("JOINT TRAINING FROM SCRATCH")
        folder_path = 'jnt_fr_scrcth' + timestamp
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)

    #set up the joint optimiser
    if opt is None: #TODO separate optim function to reduce code, maybe pass params?
        #set to branchynet default
        lr = 0.001 #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]

        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    best_val_loss = [[1.0,1.0], ''] #TODO make sure list size matches num of exits
    for epoch in range(joint_epochs):
        model.train()
        print("starting epoch:", epoch+1, end="... ", flush=True)
        train_loss = [0.0,0.0]
        correct_count = [0,0]
        #training loop
        for xb, yb in train_dl:
            results = model(xb)

            losses = [weighting * loss_f(res, yb)
                        for weighting, res in zip(model.exit_loss_weights,results)]

            opt.zero_grad()
            #backward
            for loss in losses[:-1]: #ee losses need to keep graph
                loss.backward(retain_graph=True)
            losses[-1].backward() #final loss, graph not required
            opt.step()

            for i,_ in enumerate(train_loss):
                train_loss[i]+=losses[i].item()
                correct_count[i]+=get_num_correct(results[i], yb)

        tr_loss_avg = [loss/(len(train_dl)*batch_size) for loss in train_loss]
        t1acc = [corr/(len(train_dl)*batch_size) for corr in correct_count]

        #validation
        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)


        val_loss_avg = valid_losses / (len(valid_dl)*batch_size)
        print("t loss:", tr_loss_avg, "t1acc: ", t1acc, " v loss:", val_loss_avg)
        if dat_norm:
            prefix = "dat_norm-"+prefix
        savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt)

        el_total=0.0
        bl_total=0.0
        #TODO add in the weighting for the losses
        for exit_loss, best_loss,l_w in zip(val_loss_avg,best_val_loss[0],model.exit_loss_weights):
            el_total+=exit_loss*l_w
            bl_total+=best_loss*l_w
        if el_total < bl_total:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
        #NOTE not using total loss now, using first exit loss
        #if val_loss_avg[0] < best_val_loss[0][0]:
        #    best_val_loss[0] = val_loss_avg
        #    best_val_loss[1] = savepoint

    print("BEST* VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    #return best val loss path link
    return best_val_loss[1],savepoint

def test(model,test_dl, loss_f=nn.CrossEntropyLoss(),ee_net=False):
    #if ee_net is false then there is only one exit to the network
    #FIXME make this work for one exit

    #setting top1acc threshold for exiting (final exit set to 0)
    e_thr_top1 = [0.995,0]
    e_bins_top1 = [0,0]
    assert len(e_thr_top1) == len(e_bins_top1),"threshold & bin mismatch - top1accuracy"
    #setting entropy threshold for exiting (final exit set to LARGE)
    e_thr_entr = [0.025,1000000]
    e_bins_entr = [0,0]
    assert len(e_thr_entr) == len(e_bins_entr),"threshold & bin mismatch - entropy"

    #run fast inference on test set
    #for the time being, run on one sample at a time
    sample_total = len(test_dl)
    print("test length:",sample_total)
    test_iter = iter(test_dl)
    model.eval()

    correct_count_top1 = [0,0]
    correct_count_entr = [0,0]
    with torch.no_grad():
        for s_idx in range(sample_total): #small test to check exit function
            #print("sample index",s_idx)
            xb, yb = next(test_iter)

            exits = model(xb)
            #for i,exit in enumerate(exits):
            #    #print("raw exit {}: {}".format(i, exit))
            #    softmax = nn.functional.softmax(exit,dim=-1)
            #    #print("softmax exit {}: {}".format(i, softmax))
            #    sftmx_max = torch.max(softmax)
            #    print("exit {} max softmax: {}".format(i, sftmx_max))
            #    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
            #    print("exit {} entropy: {}".format(i, entr))
            #    #print("exit CE loss: {}".format(loss_f(exit,yb)))

            for i,(exit,thr) in enumerate(zip(exits,e_thr_top1)):
                softmax = nn.functional.softmax(exit,dim=-1)
                sftmx_max = torch.max(softmax)
                if sftmx_max > thr:
                    #print("top1 exited at exit {}".format(i))
                    e_bins_top1[i]+=1
                    correct_count_top1[i]+=get_num_correct(exit,yb)
                    break

            for i,(exit,thr) in enumerate(zip(exits,e_thr_entr)):
                softmax = nn.functional.softmax(exit,dim=-1)
                entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                if entr < thr:
                    #print("entr exited at exit {}".format(i))
                    e_bins_entr[i]+=1
                    correct_count_entr[i]+=get_num_correct(exit,yb)
                    break


    assert sample_total == sum(e_bins_top1), f"too many or too few exits - top1 {e_bins_top1}"
    assert sample_total == sum(e_bins_entr), f"too many or too few exits - entr {e_bins_entr}"

    #working out percentage exit
    #top1 method
    t1pc = [(bins*100)/sample_total for bins in e_bins_top1]

    #entropy method
    entpc = [(bins*100)/sample_total for bins in e_bins_entr]

    #exit accuracies for different methods
    t1acc = [corr/ex_tot for corr,ex_tot in zip(correct_count_top1,e_bins_top1)]
    entra = [corr/ex_tot for corr,ex_tot in zip(correct_count_entr,e_bins_entr)]

    t1acc_tot = sum(correct_count_top1)/sample_total
    entra_tot = sum(correct_count_entr)/sample_total

    #test_losses = np.sum(np.array(
    #        [[loss_f(exit, yb) for exit in model(xb)]
    #            for xb, yb in valid_dl]), axis=0)

    #TODO save test stats along with link to saved model
    return t1pc, entpc, sample_total,t1acc,entra,t1acc_tot,entra_tot

def train_n_test(args):
    #shape testing
    #print(shape_test(model, [1,28,28], [1])) #output is not one hot encoded

    ee_net = False # set to true when dealing with a branching network
    #set up the model specified in args
    if args.model_name == 'lenet':
        model = Lenet()
    elif args.model_name == 'testnet':
        model = Testnet()
    elif args.model_name == 'brnfirst':
        model = Testnet()
    elif args.model_name == 'brnsecond':
        model = Testnet()
    elif args.model_name == 'b_lenet':
        model = B_Lenet()
        ee_net = True
    else:
        raise NameError("Model not supported")
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
        datacoll = MNISTDataColl(batch_size_test=batch_size_test)
        #TODO make use of profiling split
        notes_path = os.path.join(os.path.split(args.trained_model_path)[0],'notes.txt')
        save_path = args.trained_model_path

    else:
        #get data and load if not already exiting - MNIST for now
        batch_size_train = 500 #training bs in branchynet
        validation_split = 0.2
        #sort into training, and test data
        datacoll = MNISTDataColl(batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,normalise=normalise,v_split=validation_split)
        train_dl = datacoll.get_train_dl()
        valid_dl = datacoll.get_valid_dl()
        print("Got training data, batch size:",batch_size_train)

        #start training loop for epochs - at some point add recording points here
        path_str = 'outputs/'
        print("backbone epochs: {} joint epochs: {}".format(args.bb_epochs, args.jt_epochs))

        if ee_net:
            save_path,last_path = train_joint(model, train_dl, valid_dl, batch_size_train, path_str,
                    backbone_epochs=args.bb_epochs,joint_epochs=args.jt_epochs, loss_f=loss_f,
                    pretrain_backbone=True,dat_norm=normalise)
        else:
            #provide optimiser for non ee network
            lr = 0.001 #Adam algo - step size alpha=0.001
            #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
            exp_decay_rates = [0.99, 0.999]
            opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

            path_str = f'outputs/bb_only/'
            save_path,last_path = train_backbone(model, train_dl, valid_dl,
                    batch_size=batch_size, save_path=path_str, epochs=args.bb_epochs,
                    loss_f=loss_f, opt=opt, dat_norm=normalise)

        #save some notes about the run
        notes_path = os.path.join(os.path.split(save_path)[0],'notes.txt')
        with open(notes_path, 'w') as notes:
            notes.write("bb epochs {}, jt epochs {}\n".format(args.bb_epochs, args.jt_epochs))
            notes.write("Training batch size {}, Test batchsize {}\n".format(batch_size_train,
                                                                           batch_size_test))
            notes.write("model training exit weights:"+str(model.exit_loss_weights))
            notes.write("Path to last model:"+str(last_path)+"\n")
        notes.close()

        #TODO sort out graph of training data
        #separate graphs for pre training and joint training

    test_dl = datacoll.get_test_dl()
    #once trained, run it on the test data
    top1_pc,entropy_pc,test_size,top1acc,entracc,t1_tot_acc,ent_tot_acc =\
            test(model,test_dl,loss_f,ee_net)
    #get percentage exits and avg accuracies, add some timing etc.
    print("top1 exit %s {},  entropy exit %s {}".format(top1_pc, entropy_pc))
    print("SPLIT: top1 exit acc % {}, entropy exit acc % {}".format(top1acc, entracc))
    print("COMBINED: top1 acc % {}, entr acc % {}".format(t1_tot_acc,ent_tot_acc))

    with open(notes_path, 'a') as notes:
        notes.write("\nTesting results:\n")
        notes.write("Test sample size: {}\n".format(test_size))
        notes.write("top1 exit %s {}, entropy exit %s {}\n".format(top1_pc, entropy_pc))
        notes.write("best* model "+save_path)
        notes.write("SPLIT: top1 exit acc % {}, entropy exit acc % {}\n".format(top1acc, entracc))
        notes.write("COMBINED: top1 acc % {}, entr acc % {}\n".format(t1_tot_acc,ent_tot_acc))

        if args.run_notes is not None:
            notes.write(args.run_notes)
    notes.close()

    #be nice to have comparison against pytorch pretrained LeNet from pytorch

if __name__ == "__main__":
    train_n_test()
