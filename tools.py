import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt

################################
###### Data set functions ######
################################

class DataColl:
    def __init__(self,
            batch_size_train=64,
            batch_size_valid=64,
            batch_size_test=1,
            normalise=False,
            k_cv=None,
            v_split=None,
            num_workers=1
            ):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        #bool to normalise training set or not
        self.normalise_train = normalise
        self.num_workers=num_workers
        #how many equal partitions of the training data for k fold CV, e.g. 5
        self.k_cross_validation = k_cv
        if self.k_cross_validation is not None:
            print("NO K_CV YET")
        #faction of training date for validation for single train/valid split, e.g. 0.2
        self.validation_split = v_split

        assert ((k_cv is None) or (v_split is None)), "only one V type, or none at all"
        self.has_valid = True if v_split is not None or k_cv is not None else False
        self.single_split = True if v_split is not None else False

        self._load_sets()

        #torchvision datasets (dataloader precursors)
        self.train_set = None
        self.valid_set = None
        #dataloaders for single split
        self.train_dl = None
        self.valid_dl = None
        #generate data loaders
        self.get_train_dl()
        if self.has_valid and self.single_split:
            self.get_valid_dl()

        self.test_dl = None
        self.get_test_dl()
        return
    def _load_sets(self):
        self.tfs = None
        #full training set, no normalisation
        self.full_train_set = None
        #full testing set
        self.full_test_set = None
        NameError("template class, no loading")

    #####  single split methods  #####
    def gen_train(self): #gen training sets, normalised or valid split defined here
        if self.normalise_train:
            print("WARNING: Normalising data set")
            raise NameError("no normalising till fixed")
            #calc mean and stdev
            norm_dl = DataLoader(self.full_train_set, batch_size=len(self.full_train_set),
                    num_workers=self.num_workers)
            norm_data = next(iter(norm_dl))
            mean = norm_data[0].mean()
            std = norm_data[0].std()
            print("Dataset mean:{} std:{}".format(mean, std))
            # set new transforms
            tfs_norm = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
            # normalised set
            self.full_train_set = torchvision.datasets.MNIST('../data/mnist',
                    download=True, train=True, transform=tfs_norm)

        if self.validation_split is not None:
            valid_len = int(len(self.full_train_set)*self.validation_split)
            train_len = len(self.full_train_set) - valid_len
            self.train_set,self.valid_set = random_split(self.full_train_set,
                                            [train_len,valid_len])

    def get_train_dl(self,force=False):
        if force: #force will regenerate the dls - new random dist or changing split
            self.gen_train()
            self.valid_dl = None #reset valid dl in case force not called here
            self.train_dl = DataLoader(self.train_set, batch_size=self.batch_size_train,
                        drop_last=True, shuffle=True, num_workers=self.num_workers)
        else:
            if self.train_set is None:
                self.gen_train()
            elif self.train_dl is None:
                if self.single_split:
                    self.train_dl = DataLoader(self.train_set, batch_size=self.batch_size_train,
                            drop_last=True, shuffle=True, num_workers=self.num_workers)
                else:
                    self.train_dl = DataLoader(self.full_train_set, batch_size=self.
                            batch_size_train,drop_last=True, shuffle=True,
                            num_workers=self.num_workers)
        #returns training set
        return self.train_dl

    def gen_valid(self):
        assert self.validation_split is not None, "NO validation split specified"
        self.gen_train()

    def get_valid_dl(self):
        if self.valid_set is None:
            self.gen_valid()
        elif self.valid_dl is None:
            self.valid_dl = DataLoader(self.valid_set, batch_size=self.batch_size_train,
                    drop_last=True, shuffle=True, num_workers=self.num_workers)
        #returns validation split
        return self.valid_dl

    #####  test methods  #####
    def gen_test(self): #NOTE might become more complex in future
        assert self.full_test_set is not None, "Something wrong with test gen"

    def get_test_dl(self):
        self.gen_test() #NOTE only assertion for now
        if self.test_dl is None:
            self.test_dl = DataLoader(self.full_test_set, batch_size=self.batch_size_test,
                    drop_last=True, shuffle=True, num_workers=self.num_workers)
        return self.test_dl

class MNISTDataColl(DataColl):
    def _load_sets(self):
        #child version of function, MNIST specific
        #standard transform for MNIST
        self.tfs = transforms.Compose([transforms.ToTensor()])
        #full training set, no normalisation
        self.full_train_set = torchvision.datasets.MNIST('../data/mnist',
            download=True, train=True, transform=self.tfs)
        #full testing set
        self.full_test_set = torchvision.datasets.MNIST('../data/mnist',
                download=True, train=False, transform=self.tfs)

class CIFAR10DataColl(DataColl):
    def _load_sets(self):
        #child version of function, CIFAR10 specific
        #standard transform for CIFAR10
        self.tfs = transforms.Compose([transforms.ToTensor()])
        #full training set, no normalisation
        self.full_train_set = torchvision.datasets.CIFAR10('../data/cifar10',
            download=True, train=True, transform=self.tfs)
        #full testing set
        self.full_test_set = torchvision.datasets.CIFAR10('../data/cifar10',
                download=True, train=False, transform=self.tfs)

class CIFAR100DataColl(DataColl):
    def _load_sets(self):
        #child version of function, CIFAR100 specific
        #standard transform for CIFAR100
        self.tfs = transforms.Compose([transforms.ToTensor()])
        #full training set, no normalisation
        self.full_train_set = torchvision.datasets.CIFAR100('../data/cifar100',
            download=True, train=True, transform=self.tfs)
        #full testing set
        self.full_test_set = torchvision.datasets.CIFAR100('../data/cifar100',
                download=True, train=False, transform=self.tfs)

################################
######   Stat functions   ######
################################
class Tracker: #NOTE need to change add_ methods if more avgs required
    def __init__(self,
            batch_size,
            bins=1,
            set_length=None
            ):
        #init vars
        self._init_vars(batch_size,bins,set_length)
        self.avg_vals = None

    def _init_vars(self,
            batch_size,
            bins,
            set_length=None):
        self.batch_size = batch_size #NOTE if batch size differs from used then answer incorrect
        #if set_length is None:
        #    print("WARNING, no set length specified, using accumulated number")
        self.set_length = set_length
        #init bins
        self.bin_num = bins
        self.val_bins = np.zeros(bins,dtype=np.float64)
        self.set_length_accum = np.zeros(bins,dtype=np.int)

    ### functions to use ###
    def add_val(self,value,bin_index=None): #adds val(s) for single iteration
        if isinstance(value,list):
            assert len(value) == self.bin_num, "val list length mismatch {} to {}".format(
                                                                len(value),self.bin_num)
            # NOTE having to add loop to get around cpu/gpu mismatch
            for idx,v in enumerate(value):
                self.val_bins[idx] = self.val_bins[idx] + v
            self.set_length_accum = self.set_length_accum + 1 #NOTE  mul by bs in the avg
            return

        if bin_index is None and self.bin_num == 1:
            bin_index = 0
        elif bin_index is not None:
            assert bin_index < self.bin_num, "index out of range for adding individual loss"
        self.val_bins[bin_index] += value
        self.set_length_accum[bin_index] += 1#NOTE  mul by bs in the avg
        return

    def add_vals(self,val_array): #list of lists
        # [[bin0,bin1,...,binN],[bin0,bin1,...,binN],...,lossN]
        #convert to numpy array and sum across val dimension
        self.set_length_accum = np.full((self.bin_num,), len(val_array))
        self.val_bins = np.sum(np.array(val_array), axis=0)
        assert self.val_bins.shape[0] == self.bin_num,\
                f"bin mismatch {self.bin_num} with incoming array{self.val_bins.shape}"

    def reset_tracker(self,batch_size=None,bins=None,set_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if bins is None:
            bins = self.bin_num
        if set_length is None:
            set_length = self.set_length
        #print("Resetting tracker. batch size:{} bin number:{} set length:{}".format(
        #                                    batch_size,bins,set_length))
        self._init_vars(batch_size,bins,set_length)

    ### stat functions ###
    def get_avg(self,return_list=False): #mean average
        if self.set_length is not None:
            #for i,length in enumerate(self.set_length_accum):
            #    assert self.set_length == length,\
            #    "specified set length:{} differs from accumulated:{} at index {}\
            #    (might be more)".format(self.set_length,length,i)
            #use set_length
            divisor = self.set_length * self.batch_size
        else:
            #use accumulated values
            divisor = self.set_length_accum * self.batch_size

        self.avg_vals = self.val_bins / divisor
        if return_list:
            return self.avg_vals.tolist()
        return self.avg_vals

#loss (validation, testing)
class LossTracker(Tracker): #NOTE need to change add_ methods if more avgs required
    def __init__(self,
            batch_size,
            bins=1,
            set_length=None
            ):
        #init vars
        super().__init__(batch_size,bins,set_length)

    ### functions to use ###
    def add_loss(self,value,bin_index=None): #adds loss(es) for single iteration
        super().add_val(value,bin_index)

    def add_losses(self,val_array): #list of lists
        super().add_vals(val_array)

#accuracy (number correct / number classified)
class AccuTracker(Tracker):
    def __init__(self,
            batch_size,
            bins=1,
            set_length=None
            ):
        #init vars
        super().__init__(batch_size,bins,set_length)
    def _init_vars(self, #NOTE can you overloaded parent function
            batch_size,
            bins,
            set_length=None):
        self.batch_size = batch_size #NOTE if batch size differs from used then answer incorrect
        #if set_length is None:
        #    print("WARNING, no set length specified, using accumulated number")
        self.set_length = set_length
        #init bins
        self.bin_num = bins
        self.val_bins = np.zeros(bins,dtype=np.int)
        self.set_length_accum = np.zeros(bins,dtype=np.int)
    def get_num_correct(self, preds, labels):
        #predictions from model (not one hot), correct labels
        return preds.argmax(dim=1).eq(labels).sum().item()

    ### functions to use ###
    def update_correct(self,result,label,bin_index=None): #for single iteration
        if bin_index is None and len(result) > 1 and self.bin_num > 1:
            count = [self.get_num_correct(val,label) for val in result]
        else:
            if isinstance(result, list):
                count = self.get_num_correct(result[0],label)
            else:
                count = self.get_num_correct(result,label)
        super().add_val(count,bin_index)

    def update_correct_list(self,res_list,lab_list=None): #list of lists of lists
        # [[bin0,bin1,...,binN],[bin0,bin1,...,binN],...,sampN], [label0,...labelN]
        if lab_list is not None:
            assert len(res_list) == len(lab_list), "AccuTracker: sample size mismatch"
            super().add_vals([[self.get_num_correct(res,label) for res in results]
                                        for label,results in zip(lab_list,res_list)])
        else:
            super().add_vals(res_list)
    def get_accu(self,return_list=False):
        return super().get_avg(return_list)

#calibration (measuring confidence, correcting over/under confidence)


################################
######   Save functions   ######
################################
def save_model(model, path, file_prefix='', seed=None, epoch=None, opt=None,
        tloss=None,vloss=None,taccu=None,vaccu=None):
    #TODO add saving for inference only
    #TODO add bool to save as onnx - remember about fixed bs
    #saves the model in pytorch format to the path specified
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    filenm = file_prefix + '-' + timestamp
    save_dict ={'timestamp': timestamp,
                'model_state_dict': model.state_dict()
                }

    if seed is not None:
        save_dict['seed'] = seed
    if epoch is not None:
        save_dict['epoch'] = epoch
        filenm += f'{epoch:03d}'
    if opt is not None:
        save_dict['opt_state_dict'] = opt.state_dict()
    if tloss is not None:
        save_dict['tloss'] = tloss
    if vloss is not None:
        save_dict['vloss'] = vloss
    if taccu is not None:
        save_dict['taccu'] = taccu
    if vaccu is not None:
        save_dict['vaccu'] = vaccu
    if hasattr(model,'exit_loss_weights'):
        save_dict['exit_loss_weights'] = model.exit_loss_weights

    if not os.path.exists(path):
        os.makedirs(path)

    filenm += '.pth'
    file_path = os.path.join(path, filenm)

    torch.save(save_dict, file_path)
    print("Saved to:", file_path)
    return file_path

def load_model(model, path):
    #TODO add "warmstart" - partial reloading of model, useful for backbone pre_training
    #loads the model from the path specified
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    #TODO optionals
    #opt.load_state_dict(checkpoint['opt_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

################################
######  Helper functions  ######
################################
def probe_params(model):
    #probe params to double check only backbone run
    print("backbone 1st conv")
    print([param for param in model.backbone[0].parameters()])
    print("backbone last linear")
    print([param for param in model.exits[-1].parameters()])
    print("exit 1")
    print([param for param in model.exits[0].parameters()])

#checks the shape of the input and output of the model
def shape_test(model, dims_in, dims_out, loss_function=nn.CrossEntropyLoss()):
    rand_in = torch.rand(tuple([1, *dims_in]))
    rand_out = torch.rand(tuple([*dims_out])).long()

    model.eval()
    with torch.no_grad():
        results = model(rand_in)
        if isinstance(results, list):
            losses = [loss_function(res, rand_out) for res in results ]
        else:
            losses = [loss_function(results, rand_out)]
    return losses

################################
######  HW datagen funcs  ######
################################
def save_batch():
    # TODO:
    # target dir - with date, make input
    # data set - with date, make input
    target_dir="./IMAGES_1024f/"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    dc = MNISTDataColl(256,256,1)

    test_dl = dc.get_test_dl()

    bs = 1024

    idx=0
    for xb,yb in test_dl:
        #print("shape:",xb.shape)
        #save_image(xb, './IMAGES/img{:05d}.png'.format(idx))
        np.save(target_dir+('imgf{:05d}.npy'.format(idx)), xb)
        idx+=1
        if idx >= bs:
            break
    return

def exit_pc(ee_pc=0.70, bs=1024):
    print("Generating hw test set.")
    # hw test values
    # ee_pc: early exit percentage
    # bs: batch size

    target_dir="./IMAGES_ee-pc{}_bs{}/".format(int(ee_pc*100), bs)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    #load examples
    ee_eg = np.load("./EE_EXAMPLE.npy")
    le_eg = np.load("./LE_EXAMPLE.npy")

    le_pc = 1-ee_pc
    hw_test = random.choices([ee_eg, le_eg],
            weights=[ee_pc,le_pc], k=bs)
    # save values
    for idx, sample in enumerate(hw_test):
        np.save(target_dir+('img{:05d}.npy'.format(idx)) ,sample)
    return
