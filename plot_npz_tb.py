import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_npz(filename, export=None, act_quant_line=None):
    data = dict(np.load(filename))
    if 'num_trials' in data:
        del data['num_trials']
    plot_data(data, export, act_quant_line)


def plot_tb(filename, export=None, act_quant_line=None):
    from eegnet_run import _prepare_scalar_array_from_tensorboard as prepare_tb_array
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(filename)
    ea.Reload()
    data = {key: prepare_tb_array(ea, key) for key in ea.Tags()['scalars']}
    plot_data(data, export, act_quant_line)


def plot_data(data, export=None, act_quant_line=None):
    # decide for each key to which plot it should belong
    loss_plot = {}
    acc_plot = {}

    n_epochs = None

    for name, array in data.items():
        if n_epochs is None:
            n_epochs = len(array)
        else:
            assert len(array) == n_epochs, f"{name} has length {len(array)} but should be {n_epochs}"

        l_name = name.lower()
        if 'metric' in l_name or 'acc' in l_name or 'accuracy' in l_name:
            acc_plot[name] = array
        elif 'loss' in l_name:
            loss_plot[name] = array
        elif l_name == 'learning_rate':
            pass
        else:
            # ask user to which plot it should be added
            choice = input(f"Where to put {name}? [b]oth, [l]oss, [a]ccuracy, [N]one? > ")
            choice = choice.lower() if choice else 'n'
            assert choice in ['b', 'l', 'a', 'n']
            if choice in ['b', 'l']:
                loss_plot[name] = array
            if choice in ['b', 'a']:
                acc_plot[name] = array

    generate_figure(loss_plot, acc_plot, n_epochs, export, act_quant_line)


def generate_figure(loss_plot, acc_plot, n_epochs, export=None, act_quant_line=None):

    # make sure that the environment variables are set (to hide the unnecessary output)
    if "XDG_RUNTIME_DIR" not in os.environ:
        tmp_dir = "/tmp/runtime-eegnet"
        os.environ["XDG_RUNTIME_DIR"] = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            os.chmod(tmp_dir, 700)

    # prepare data
    x = np.array(range(1, n_epochs + 1))

    # prepare the plot
    fig = plt.figure(figsize=(20, 10))

    # do loss figure
    loss_subfig = fig.add_subplot(121)
    add_subplot(loss_plot, x, loss_subfig, "Loss", "upper center", act_quant_line)

    # do accuracy figure
    acc_subfig = fig.add_subplot(122)
    add_subplot(acc_plot, x, acc_subfig, "Accuracy", "lower center", act_quant_line)

    # save the image
    if export is None:
        plt.show()
    else:
        fig.savefig(export, bbox_inches='tight')

    # close
    plt.close('all')


def add_subplot(data, x, subfig, title, legend_pos=None, act_quant_line=None):
    plt.grid()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    additional_axis = []
    lines = []

    if act_quant_line is not None:
        lines.append(plt.axvline(x=act_quant_line, label='Activation Quantization', color=colors[2]))

    for i, key in enumerate(data.keys()):
        if key.startswith('train_'):
            new_lines = subfig.plot(x, data[key], label=key, color=colors[0])
        elif key.startswith('valid_'):
            new_lines = subfig.plot(x, data[key], label=key, color=colors[1])
        else:
            tmp_axis = subfig.twinx()
            tmp_axis.set_ylabel(key)
            new_lines = tmp_axis.plot(x, data[key], label=key, color=colors[i+3])
            additional_axis.append(tmp_axis)
        lines += new_lines

    for i, axis in enumerate(additional_axis):
        axis.spines['right'].set_position(('axes', 1 + i * 0.15))
        if i > 0:
            axis.set_frame_on(True)
            axis.patch.set_visible(False)

    subfig.set_title(title)
    subfig.set_xlabel("Epoch")

    labels = [l.get_label() for l in lines]
    last_ax = additional_axis[-1] if additional_axis else subfig
    last_ax.legend(lines, labels, frameon=True, framealpha=1, facecolor='white', loc=legend_pos)

    return len(additional_axis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='filename of the data', nargs=1)
    parser.add_argument('-t', '--tensorboard', help='Data is of tensorboard format',
                        action='store_true')
    parser.add_argument('-n', '--numpy', help='Data is of numpy npz format',
                        action='store_true')
    parser.add_argument('-e', '--export', help='export plot to specified file', type=str)
    parser.add_argument('--act_quant_line', help='position of vertical line', type=int)

    args = parser.parse_args()

    # if both tensorboard and numpy are not set, infer the type by the file ending
    filename = args.file[0]
    if not args.tensorboard and not args.numpy:
        if 'events.out.tfevents' in filename:
            args.tensorboard = True
        elif filename.endswith('.npz'):
            args.numpy = True
        else:
            raise RuntimeError(f'Cannot automatically detect type of the file: {args.file}')

    if args.tensorboard:
        plot_tb(filename, args.export, args.act_quant_line)
    elif args.numpy:
        plot_npz(filename, args.export, args.act_quant_line)
    else:
        raise RuntimeError()
