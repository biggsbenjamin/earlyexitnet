"""
Class for testing early exit and normal CNNs.
Includes requires accuracy and loss trackers from tools
"""

# importing trackers for loss + accuracy, and generic tracker for exit distribution
from earlyexitnet.tools import Tracker, LossTracker, AccuTracker

class Tester:
    def __init__(self,model,test_dl,loss_f=nn.CrossEntropyLoss(),exits=2,
            top1acc_thresholds=[],entropy_thresholds=[]):
        self.model=model
        self.test_dl=test_dl
        self.loss_f=loss_f
        self.exits=exits
        self.sample_total = len(test_dl)
        self.top1acc_thresholds = top1acc_thresholds
        self.entropy_thresholds = entropy_thresholds

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
        with torch.no_grad():
            for xb,yb in self.test_dl:
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
        with torch.no_grad():
            for xb,yb in self.test_dl:
                res = self.model(xb)
                self.accu_track_totl.update_correct(res,yb)

    def debug_values(self):
        self.model.eval()
        with torch.no_grad():
            for xb,yb in test_dl:
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
