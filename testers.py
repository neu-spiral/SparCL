import numpy as np


def test_sparsity(model, column=True, channel=True, filter=True, kernel=False):

    # --------------------- total sparsity --------------------
    total_zeros = 0
    total_nonzeros = 0
    layer_cont = 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            print("(empty/total) weights of {}({}) is: ({}/{}). irregular sparsity is: {:.4f}".format(
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

    # --------------------- column sparsity --------------------
    if(column):

        total_column = 0
        total_empty_column = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                weight2d = weight.reshape(weight.shape[0], -1)
                column_num = weight2d.shape[1]

                empty_column = np.sum(np.sum(np.absolute(weight2d.cpu().detach().numpy()), axis=0) == 0)
                print("(empty/total) column of {}({}) is: ({}/{}). column sparsity is: {:.4f}".format(
                    name, layer_cont, empty_column, weight.size()[1] * weight.size()[2] * weight.size()[3],
                                        empty_column / column_num))

                total_column += column_num
                total_empty_column += empty_column
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of column: {}, empty-column: {}, column sparsity is: {:.4f}".format(
            total_column, total_empty_column, total_empty_column / total_column))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- channel sparsity --------------------
    if (channel):

        total_channels = 0
        total_empty_channels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                empty_channels = 0
                channel_num = weight.size()[1]

                for i in range(channel_num):
                    if np.sum(np.absolute(weight[:, i, :, :].cpu().detach().numpy())) == 0:
                        empty_channels += 1
                print("(empty/total) channel of {}({}) is: ({}/{}) ({}). channel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_channels, weight.size()[1], weight.size()[1]-empty_channels, empty_channels / channel_num))

                total_channels += channel_num
                total_empty_channels += empty_channels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of channels: {}, empty-channels: {}, channel sparsity is: {:.4f}".format(
            total_channels, total_empty_channels, total_empty_channels / total_channels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- filter sparsity --------------------
    if(filter):

        total_filters = 0
        total_empty_filters = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                empty_filters = 0
                filter_num = weight.size()[0]

                for i in range(filter_num):
                    if np.sum(np.absolute(weight[i, :, :, :].cpu().detach().numpy())) == 0:
                        empty_filters += 1
                print("(empty/total) filter of {}({}) is: ({}/{}) ({}). filter sparsity is: {:.4f} ({:.4f})".format(
                    name, layer_cont, empty_filters, weight.size()[0], weight.size()[0]-empty_filters, empty_filters / filter_num, 1-(empty_filters / filter_num)))

                total_filters += filter_num
                total_empty_filters += empty_filters
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of filters: {}, empty-filters: {}, filter sparsity is: {:.4f}".format(
            total_filters, total_empty_filters, total_empty_filters / total_filters))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

    # --------------------- kernel sparsity --------------------
    if(kernel):

        total_kernels = 0
        total_empty_kernels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                shape = weight.shape
                npWeight = weight.cpu().detach().numpy()
                weight3d = npWeight.reshape(shape[0], shape[1], -1)

                empty_kernels = 0
                kernel_num = weight.size()[0] * weight.size()[1]

                for i in range(weight.size()[0]):
                    for j in range(weight.size()[1]):
                        if np.sum(np.absolute(weight3d[i, j, :])) == 0:
                            empty_kernels += 1
                print("(empty/total) kernel of {}({}) is: ({}/{}) ({}). kernel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_kernels, kernel_num, kernel_num-empty_kernels, empty_kernels / kernel_num))

                total_kernels += kernel_num
                total_empty_kernels += empty_kernels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of kernels: {}, empty-kernels: {}, kernel sparsity is: {:.4f}".format(
            total_kernels, total_empty_kernels, total_empty_kernels / total_kernels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")
    return comp_ratio, total_sparsity



