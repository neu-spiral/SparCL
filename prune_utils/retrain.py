
import torch
import logging
import sys
import os
import numpy as np
import argparse
import time
import random
import copy
from . import utils_pr
# from .admm import weight_growing, weight_pruning, ADMM
from .utils_pr import weight_pruning, weight_growing

def prune_parse_arguments(parser):
    parser.add_argument('--retrain-mask-pattern', type=str, default='weight',
                    help="retrain mask pattern")
    parser.add_argument('--sp-update-init-method', type=str, default='zero',
                        help="mask update initialization method")
    parser.add_argument('--sp-mask-update-freq', type=int, default=5,
                        help="how many epochs to update sparse mask")
    parser.add_argument('--sp-lmd', type=float, default=0.5,
                        help="importance coefficient lambda")
    parser.add_argument('--retrain-mask-sparsity', type=float, default=-1.0,
                    help="sparsity of a retrain mask, used when retrain-mask-pattern is set to NOT being 'weight' ")
    parser.add_argument('--retrain-mask-seed', type=int, default=None,
                    help="seed to generate a random mask")
    parser.add_argument('--sp-prune-before-retrain', action='store_true',
                        help="Prune the loaded model before retrain, in case of loading a dense model")
    parser.add_argument('--output-compressed-format', action='store_true',
                        help="output compressed format")
    parser.add_argument("--sp-grad-update", action="store_true",
                        help="enable grad update when training in random GaP")
    parser.add_argument("--sp-grad-decay", type=float, default=0.98,
                        help="The decay number for gradient")
    parser.add_argument("--sp-grad-restore-threshold", type=float, default=-1,
                        help="When the decay")
    parser.add_argument("--sp-global-magnitude", action="store_true",
                        help="Use global magnitude to prune models")
    parser.add_argument('--sp-pre-defined-mask-dir', type=str, default=None,
                        help="using another sparse model to init sparse mask")

    parser.add_argument('--upper-bound', type=str, default=None,
                        help="using another sparse model to init sparse mask")
    parser.add_argument('--lower-bound', type=str, default=None,
                        help="using another sparse model to init sparse mask")
    parser.add_argument('--mask-update-decay-epoch', type=str, default=None,
                        help="using another sparse model to init sparse mask")


class SparseTraining(object):
    def __init__(self, args, model, logger=None, pre_defined_mask=None, seed=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.pattern = self.args.retrain_mask_pattern
        self.pre_defined_mask = pre_defined_mask # as model's state_dict
        self.sparsity = self.args.retrain_mask_sparsity
        self.seed = self.args.retrain_mask_seed
        self.sp_mask_update_freq = self.args.sp_mask_update_freq
        self.update_init_method = self.args.sp_update_init_method
        self.seq_gap_layer_indices = None

        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            self.logger = logging.getLogger("pruning")
        else:
            self.logger = logger

        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.masks = {}
        self.gradient_masks = {}
        self.masked_layers = {}
        self.configs, self.prune_ratios = utils_pr.load_configs(model, args.sp_config_file, self.logger)

        if "masked_layers" in self.configs:
            self.masked_layers = self.configs['masked_layers']
        else:
            for name, W in (self.model.named_parameters()):
                self.masked_layers[utils_pr.canonical_name(name)] = None


        if "fixed_layers" in self.configs:
            self.fixed_layers = self.configs['fixed_layers']
        else:
            self.fixed_layers = None
        self.fixed_layers_save = {}

        if self.args.upper_bound != None:
            self.upper_bound = self.args.upper_bound
            print("!!!!! upper_bound", self.upper_bound)
        else:
            self.upper_bound = None

        if self.args.lower_bound != None:
            self.lower_bound = self.args.lower_bound
            print("!!!!! lower_bound", self.lower_bound)
        else:
            self.lower_bound = None

        if self.args.mask_update_decay_epoch != None:
            self.mask_update_decay_epoch = self.args.mask_update_decay_epoch
        else:
            self.mask_update_decay_epoch = None

        # if "upper_bound" in self.configs:
        #     self.upper_bound = self.configs['upper_bound']
        # else:
        #     self.upper_bound = None
        # if "lower_bound" in self.configs:
        #     self.lower_bound = self.configs['lower_bound']
        # else:
        #     self.lower_bound = None
        # if "mask_update_decay_epoch" in self.configs:
        #     self.mask_update_decay_epoch = self.configs['mask_update_decay_epoch']
        # else:
        #     self.mask_update_decay_epoch = None



        self.init()

    def init(self):

        self.generate_mask(self.pre_defined_mask)


    def apply_masks(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    W.mul_((self.masks[name] != 0).type(dtype))
                    # W.data = (W * (self.masks[name] != 0).type(dtype)).type(dtype)
                    pass

    def apply_masks_on_grads(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.masks[name] != 0).type(dtype))
                    pass

    def apply_masks_on_grads_efficient(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.gradient_masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.gradient_masks[name] != 0).type(dtype))
                    pass

    def apply_masks_on_grads_mix(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.masks[name] != 0).type(dtype))
            
            for name, W in (self.model.named_parameters()):
                if name not in self.masks:  # ignore layers that do not have rho
                    continue
                # cuda_pruned_weights = None
                percent = self.args.gradient_sparse * 100
                weight_temp = np.abs(W.grad.cpu().detach().numpy(
                ))  # a buffer that holds weights with absolute values
                percentile = np.percentile(
                    weight_temp,
                    percent)  # get a value for this percentitle
                under_threshold = weight_temp < percentile
                above_threshold = weight_temp > percentile
                above_threshold = above_threshold.astype(
                    np.float32
                )  # has to convert bool to float32 for numpy-tensor conversion
                W.grad[under_threshold] = 0

                # gradient = W.grad.data
                # above_threshold, cuda_pruned_gradient = admm.weight_pruning(args, name, gradient, args.gradient_sparse)  # get sparse model in cuda
                # W.grad.data = cuda_pruned_gradient  # replace the data field in variable

                gradient = W.grad.cpu().detach().numpy()
                non_zeros = gradient != 0
                non_zeros = non_zeros.astype(np.float32)
                zero_mask = torch.from_numpy(non_zeros).cuda()
                self.gradient_masks[name] = zero_mask

    def test_mask_sparsity(self, column=False, channel=False, filter=False, kernel=False):
        
        # --------------------- total sparsity --------------------
        # comp_ratio_list = []

        total_zeros = 0
        total_nonzeros = 0
        layer_cont = 1
        mask = self.gradient_masks
        for name, weight in mask.items():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                zeros = np.sum(weight.cpu().detach().numpy() == 0)
                total_zeros += zeros
                non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
                total_nonzeros += non_zeros
                print("(empty/total) masks of {}({}) is: ({}/{}). irregular sparsity is: {:.4f}".format(
                    name, layer_cont, zeros, zeros+non_zeros, zeros / (zeros+non_zeros)))

            layer_cont += 1

        comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)
        total_sparsity = total_zeros / (total_zeros + total_nonzeros)

        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

        return comp_ratio, total_sparsity
        

    def show_masks(self, debug=False):
        with torch.no_grad():
            if debug:
                name = 'module.layer1.0.conv1.weight'
                np_mask = self.masks[name].cpu().numpy()
                np.set_printoptions(threshold=sys.maxsize)
                print(np.squeeze(np_mask)[0], name)
                return
            for name, W in self.model.named_parameters():
                if name in self.masks:
                    np_mask = self.masks[name].cpu().numpy()
                    np.set_printoptions(threshold=sys.maxsize)
                    print(np.squeeze(np_mask)[0], name)



    def update_mask(self, epoch, batch_idx):
        # a hacky way to differenate random GaP and others
        if not self.mask_update_decay_epoch:
            return
        if batch_idx != 0:
            return

        freq = self.sp_mask_update_freq

        bound_index = 0

        try: # if mask_update_decay_epoch has only one entry
            int(self.mask_update_decay_epoch)
            freq_decay_epoch = int(self.mask_update_decay_epoch)
            try: # if upper/lower bound have only one entry
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError: # if upper/lower bound have multiple entries
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity
                if epoch >= freq_decay_epoch:
                    freq *= 1
                    bound_index += 1
        except ValueError: # if mask_update_decay_epoch has multiple entries
            freq_decay_epoch = self.mask_update_decay_epoch.split('-')
            for i in range(len(freq_decay_epoch)):
                freq_decay_epoch[i] = int(freq_decay_epoch[i])

            try:
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError:
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity

                if len(freq_decay_epoch) + 1 <= len(upper_bound): # upper/lower bound num entries enough for all update
                    for decay in freq_decay_epoch:
                        if epoch >= decay:
                            freq *= 1
                            bound_index += 1
                else: # upper/lower bound num entries less than update needs, use the last entry to do rest updates
                    for idx, _ in enumerate(upper_bound):
                        if epoch >= freq_decay_epoch[idx] and idx != len(upper_bound) - 1:
                            freq *= 1
                            bound_index += 1

        lower_bound_value = float(lower_bound[bound_index])
        upper_bound_value = float(upper_bound[bound_index])

        if epoch % freq == 0:
            '''
            calculate prune_part and grow_part for sequential GaP, if no seq_gap_layer_indices specified in yaml file,
            set prune_part and grow_part to all layer specified in yaml file as random GaP do.
            '''
            prune_part, grow_part = self.seq_gap_partition()

            with torch.no_grad():
                sorted_to_prune = None
                if self.args.sp_global_magnitude:
                    total_size = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        total_size += W.data.numel()
                    to_prune = np.zeros(total_size)
                    index = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        size = W.data.numel()
                        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
                        index += size
                    sorted_to_prune = np.sort(to_prune)

                # import pdb; pdb.set_trace()
                for name, W in (self.model.named_parameters()):
                    if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                        continue

                    weight = W.cpu().detach().numpy()
                    weight_current_copy = copy.copy(weight)


                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    np_orig_mask = self.masks[name].cpu().detach().numpy()

                    print(("\n==> BEFORE UPDATE: {}: {}, {}, {}".format(name,
                                                                    str(num_nonzeros),
                                                                    str(total_num),
                                                                    str(sparsity))))

                    ############## pruning #############
                    pruned_weight_np = None
                    if name in prune_part:
                        sp_admm_sparsity_type_copy = copy.copy(self.args.sp_admm_sparsity_type)
                        sparsity_type_list = (self.args.sp_admm_sparsity_type).split("+")
                        for i in range(len(sparsity_type_list)):
                            sparsity_type = sparsity_type_list[i]
                            print("* sparsity type {} is {}".format(i, sparsity_type))
                            self.args.sp_admm_sparsity_type = sparsity_type

                            pruned_mask, pruned_weight = weight_pruning(self.args,
                                                                        self.configs,
                                                                        name,
                                                                        W,
                                                                        lower_bound_value)
                            self.args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy
                            # pruned_mask_np = pruned_mask.cpu().detach().numpy()
                            pruned_weight_np = pruned_weight.cpu().detach().numpy()

                            W.mul_(pruned_mask.cuda())


                            non_zeros_prune = pruned_weight_np != 0
                            num_nonzeros_prune = np.count_nonzero(non_zeros_prune.astype(np.float32))
                            print(("==> PRUNE: {}: {}, {}, {}".format(name,
                                                             str(num_nonzeros_prune),
                                                             str(total_num),
                                                             str(1 - (num_nonzeros_prune * 1.0) / total_num))))

                            self.masks[name] = pruned_mask.cuda()


                            if self.args.gradient_efficient:
                                new_lower_bound_value = lower_bound_value + self.args.gradient_remove
                                pruned_mask, pruned_weight = weight_pruning(self.args,
                                                                            self.configs,
                                                                            name,
                                                                            W,
                                                                            new_lower_bound_value)
                                self.gradient_masks[name] = pruned_mask.cuda()


                    ############## growing #############
                    if name in grow_part:
                        if pruned_weight_np is None: # use in seq gap
                            pruned_weight_np = weight_current_copy

                        updated_mask = weight_growing(self.args,
                                                      name,
                                                      pruned_weight_np,
                                                      lower_bound_value,
                                                      upper_bound_value,
                                                      self.update_init_method)
                        self.masks[name] = updated_mask
                        pass



    def cut_all_partitions(self, all_update_layer_name):
        # calculate the number of partitions and range
        temp1 = str(self.seq_gap_layer_indices)
        temp1 = (temp1).split('-')
        num_partition = len(temp1) + 1
        head = 0
        end = len(all_update_layer_name)
        all_range = []

        for i, indice in enumerate(temp1):
            assert int(indice) < end, "\n\n * Error, seq_gap_layer_indices must within range [0, {}]".format(end - 1)
        assert len(temp1) == len(set(temp1)), "\n\n * Error, seq_gap_layer_indices can not have duplicate element"

        for i in range(0, num_partition):
            if i == 0:
                range_i = (head, int(temp1[i]))
            elif i == num_partition - 1:
                range_i = (int(temp1[i - 1]), end)
            else:
                range_i = (int(temp1[i - 1]), int(temp1[i]))
            print(range_i)
            all_range.append(range_i)

        for j in range(num_partition):
            range_j = all_range[j]
            self.all_part_name_list.append(all_update_layer_name[range_j[0]:range_j[1]])

    def seq_gap_partition(self):
        prune_part = []
        grow_part = []

        if self.seq_gap_layer_indices is None: # Random Gap: add all layer name in prune part and grow part list
            for name, _ in self.model.named_parameters():
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                prune_part.append(name)
                grow_part.append(name)
        else: # Sequential gap One-run: partition model
            all_update_layer_name = []
            for name, _ in self.model.named_parameters():
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                all_update_layer_name.append(name)
            if not self.all_part_name_list:
                self.cut_all_partitions(all_update_layer_name) # get all partitions by name in self.all_part_name_list

            to_grow = (self.all_part_name_list).pop(0)
            to_prune = self.all_part_name_list

            for layer in to_grow:
                grow_part.append(layer)
            for part in to_prune:
                for layer in part:
                    prune_part.append(layer)

            (self.all_part_name_list).append(to_grow)

        return prune_part, grow_part



    def generate_mask(self, pre_defined_mask=None):
        masks = {}
        # import pdb; pdb.set_trace()
        if self.pattern == 'weight':


            with torch.no_grad():
                for name, W in (self.model.named_parameters()):

                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

            #for name in masks:
            #    print("Current mask includes:", name)
                    #if 'weight' in name:
                    #    print(name, (np.sum(non_zeros) + 0.0) / np.size(non_zeros) )
                #exit()



        elif self.pattern == 'random':
            if self.seed is not None:
                print("Setting the random mask seed as {}".format(self.seed))
                np.random.seed(self.seed)

            with torch.no_grad():
                # self.sparsity (args.retrain_mask_sparsity) will override prune ratio config file
                if self.sparsity > 0:
                    sparsity = self.sparsity

                    for name, W in (self.model.named_parameters()):
                        if 'weight' in name and 'bn' not in name:
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        self.masks[name] = zero_mask

                else: #self.sparsity < 0

                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        if name in self.prune_ratios:
                            # Use prune_ratio[] to indicate which layers to random masked
                            sparsity = self.prune_ratios[name]
                            '''
                            if sparsity < 0.001:
                                continue
                            '''
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()

                        self.masks[name] = zero_mask

                # # DEBUG:
                DEBUG = False
                if DEBUG:
                    for name, W in (self.model.named_parameters()):
                        m = self.masks[name].detach().cpu().numpy()
                        total_ones = np.sum(m)
                        total_size = np.size(m)
                        print( name, m.shape, (total_ones+0.0)/total_size)

                #exit()
        # TO DO
        elif self.pattern == 'regular':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    if 'weight' in name and 'bn' not in name:

                        ouputSize, inputSize = W.data.shape[0], W.data.shape[1]
                        non_zeros = np.zeros(W.data.shape)
                        non_zeros = np.squeeze(non_zeros)

                        if 'sa1.conv_blocks.0.0.weight' in name or 'sa1.conv_blocks.1.0.weight' in name or 'sa1.conv_blocks.2.0.weight' in name:
                            non_zeros[::self.args.mask_sample_rate,::] = 1

                        else:
                            non_zeros[::self.args.mask_sample_rate,::self.args.mask_sample_rate] = 1

                        non_zeros = np.reshape(non_zeros, W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()

                    else:
                        non_zeros = 1 - np.zeros(W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()
                    self.masks[name] = zero_mask
        elif self.pattern == 'global_weight':
            with torch.no_grad():
                all_w = []
                all_name = []
                print('Concatenating all weights...')
                for name, W in self.model.named_parameters():
                    if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                        continue
                    all_w.append(W.detach().cpu().numpy().flatten())
                    all_name.append(name)
                np_w = all_w[0]
                for i in range(1,len(all_w)):
                    np_w = np.append(np_w, all_w[i])

                #print(np_w.shape)
                print("All weights concatenated!")
                print("Start sorting all the weights...")
                np_w = np.sort(np.abs(np_w))
                print("Sort done!")
                L = len(np_w)
                #print(np_w)
                if self.args.retrain_mask_sparsity >= 0.0:
                    thr = np_w[int(L * self.args.retrain_mask_sparsity)]

                    for name, W in self.model.named_parameters():
                        if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                            continue


                        np_mask = np.abs(W.detach().cpu().numpy())  > thr
                        print(name, np.size(np_mask), np.sum(np_mask), float(np.sum(np_mask))/np.size(np_mask) )

                        self.masks[name] = torch.from_numpy(np_mask).cuda()

                    total_non_zero = 0
                    total_size = 0
                    with open('gw_sparsity.txt','w') as f:
                        for name, W in sorted(self.model.named_parameters()):
                            if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                                continue
                            np_mask = self.masks[name].detach().cpu().numpy()
                            sparsity = 1.0 - float(np.sum(np_mask))/np.size(np_mask)
                            if sparsity < 0.5:
                                sparsity = 0.0

                            if sparsity < 0.5:
                                total_non_zero += np.size(np_mask)
                            else:
                                total_non_zero += np.sum(np_mask)
                            total_size += np.size(np_mask)

                            f.write("{}: {}\n".format(name,sparsity))
                    print("Thr:{}".format(thr))
                    print("{},{},{}".format(total_non_zero, total_size, float(total_non_zero)/total_size))
                    exit()



        elif self.pattern == 'none':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    non_zeros = np.ones(W.data.shape)
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).cuda()
            self.masks[name] = zero_mask

        elif self.pattern == "pre_defined":
            assert pre_defined_mask is not None, "\n\n * Error, pre_defined sparse mask model must be declared!"
            with torch.no_grad():
                for name, W in pre_defined_mask.items():
                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks
