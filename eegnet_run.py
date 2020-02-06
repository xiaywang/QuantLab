import os
import shutil
import json
import sys
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import progress
from tqdm import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from main import main as quantlab_main

PROBLEM = "BCI-CompIV-2a"
TOPOLOGY = "EEGNet"
EXP_FOLDER = "logs/exp{}"
MEAS_ID = 11
INQ_CONFIG = f"measurement/M{MEAS_ID:02}.json"
BAK_CONFIG = ".config_backup.json"
MAIN_CONFIG = "config.json"
EXP_BASE = MEAS_ID * 100
EXPORT_FILE = f"logs/measurement_{MEAS_ID:02}" + "_{}.npz"

BENCHMARK = True
N_ITER = 20


def single_iter(bar=None, silent=False):
    iter_stats = np.zeros(9)
    with TestEnvironment():
        for i in range(9):
            subject = i + 1
            stats = _do_subject(subject, bar, silent)
            if not silent:
                print(f"Subject {subject}: quantized accuracy: {stats['acc']:.4f}                 ")
            iter_stats[i] = stats['acc']

    if not silent:
        print(f"Average quantized accuracy = {iter_stats.mean()}")

    return iter_stats


def benchmark():
    stats = np.zeros((N_ITER, 9))

    with tqdm(desc=f'Benchmarking Measurement {MEAS_ID:02}', total=N_ITER * 9, ascii=True) as bar:
        for i in range(N_ITER):
            iter_stats = single_iter(bar=bar, silent=True)
            stats[i, :] = iter_stats

            # store the data to make sure not to loose it
            np.savez(file=os.path.join(PROBLEM, EXPORT_FILE.format("runs")), quant=stats)

    # compute statistics
    avg_stats = stats.mean(axis=0)
    std_stats = stats.std(axis=0)

    # For the overall score, first average along all subjects.
    # For standard deviation, average all standard deviations of all subjects
    mean_avg_stats = avg_stats[:].mean()
    mean_std_stats = std_stats[:].mean()

    print(f"Total Average Accuracy: {mean_avg_stats:.4f} +- {mean_std_stats:.4f}\n")
    for i in range(0, 9):
        print(f"subject {i+1}: quantized model = {avg_stats[i]:.4f} +- {std_stats[i]:.4f}")


def _do_subject(subject, bar=None, silent=False):
    exp_id = EXP_BASE + subject

    if not silent:
        print(f"Subject {subject}: training quantized model (exp{exp_id})...\r", end='',
              flush=True)

    modification = {'treat.data.subject': subject}
    stats = _execute_quantlab(INQ_CONFIG, exp_id, modification)

    if bar is not None:
        bar.update()

    # accumulate log files
    if BENCHMARK:
        _accumulate_logs(subject, exp_id)

    return _format_stats(stats)


def _execute_quantlab(config_file, exp_id, modify_keys=None):
    # remove all the logs of the previous quantized training experiment
    log_folder = os.path.join(PROBLEM, EXP_FOLDER.format(exp_id))
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)

    # load configuration
    config = {}
    with open(os.path.join(PROBLEM, config_file)) as _fp:
        config = json.load(_fp)

    # modify keys
    for path, value in modify_keys.items():
        _set_dict_value(config, path, value)

    # store the configuration back as config.json
    if os.path.exists(os.path.join(PROBLEM, MAIN_CONFIG)):
        os.remove(os.path.join(PROBLEM, MAIN_CONFIG))
    with open(os.path.join(PROBLEM, MAIN_CONFIG), "w") as _fp:
        json.dump(config, _fp)

    # execute quantlab without output
    with open(os.devnull, 'w') as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
        stats = quantlab_main(PROBLEM, TOPOLOGY, exp_id, 'best', 'train', 10, 1, False, True)

    return stats


def _format_stats(ref_stats, quant_stats=None):
    stats = {}

    if quant_stats is None:
        for key, value in ref_stats.items():
            if key.endswith("loss"):
                stats['loss'] = value
            if key.endswith("metric"):
                stats['acc'] = value

    else:
        for key, value in ref_stats.items():
            if key.endswith("loss"):
                stats['float_loss'] = value
            if key.endswith("metric"):
                stats['float_acc'] = value
        for key, value in quant_stats.items():
            if key.endswith("loss"):
                stats['quant_loss'] = value
            if key.endswith("metric"):
                stats['quant_acc'] = value

    return stats


def _set_dict_value(d, path, value):
    keys = path.split('.')
    d_working = d
    for key in keys[:-1]:
        d_working = d_working[key]
    d_working[keys[-1]] = value


def _accumulate_logs(subject, exp_id):
    # extract name of logfile
    stats_folder = os.path.join(PROBLEM, EXP_FOLDER.format(exp_id), "stats")
    log_files = os.listdir(stats_folder)
    assert(len(log_files) == 1)
    log_file = os.path.join(stats_folder, log_files[0])

    # get eventaccumulator
    ea = EventAccumulator(log_file)
    ea.Reload()

    # load data file
    name_addon = f"data_S{subject:02}"
    data_file = os.path.join(PROBLEM, EXPORT_FILE.format(name_addon))
    if os.path.exists(data_file):
        with np.load(data_file) as data_loader:
            data = dict(data_loader)
    else:
        data = {'num_trials': 0}

    # update the data dictionary to keep the mean value
    num_trials = data['num_trials']
    for key in ea.Tags()['scalars']:
        new_arr = _prepare_scalar_array_from_tensorboard(ea, key)
        if num_trials == 0:
            # just add the data
            data[key] = new_arr
        else:
            assert(key in data)
            data[key] = (data[key] * num_trials + new_arr) / (num_trials + 1)
    data['num_trials'] += 1

    # store data back into the same file
    np.savez(data_file, **data)


def _prepare_scalar_array_from_tensorboard(ea, key, start_step=1):
    if ea.Scalars(key)[-1].step == len(ea.Scalars(key)):
        return np.array([x.value for x in ea.Scalars(key)])
    else:
        arr = np.zeros(ea.most_recent_step)
        entries = ea.Scalars(key)
        # we assume the value is zero at the beginning
        for i_entry in range(len(entries)):
            start_idx = entries[i_entry].step - start_step
            end_idx = entries[i_entry + 1].step if i_entry + 1 < len(entries) else \
                ea.most_recent_step - start_step + 1
            arr[start_idx:end_idx] = entries[i_entry].value
        return arr


class TestEnvironment():
    def __enter__(self):
        # backup config.json if it exists
        if os.path.exists(os.path.join(PROBLEM, MAIN_CONFIG)):
            os.rename(os.path.join(PROBLEM, MAIN_CONFIG),
                      os.path.join(PROBLEM, BAK_CONFIG))

        # hide progress default output
        self.devnull = open(os.devnull, 'w')
        progress.Infinite.file = self.devnull

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # remove the created config.json file
        if os.path.exists(os.path.join(PROBLEM, MAIN_CONFIG)):
            os.remove(os.path.join(PROBLEM, MAIN_CONFIG))

        # move backup back
        if os.path.exists(os.path.join(PROBLEM, BAK_CONFIG)):
            os.rename(os.path.join(PROBLEM, BAK_CONFIG),
                      os.path.join(PROBLEM, MAIN_CONFIG))

        # reenable default progress
        progress.Infinite.file = sys.stderr


if __name__ == '__main__':

    if BENCHMARK:
        benchmark()
    else:
        single_iter()
