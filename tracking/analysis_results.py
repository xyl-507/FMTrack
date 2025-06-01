import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse


trackers = []
# dataset_name = 'lasher_test' # lasot_extension_subset

parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_name', type=str, default='odtrack', help='Name of tracking method.')
parser.add_argument('--tracker_param', type=str, default='baseline_256_lasher', help='Name of config file.')
parser.add_argument('--runid', type=int, default='15', help='The run id.')
parser.add_argument('--dataset_name', type=str, default='lasher_test', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
args = parser.parse_args()

tracker_name = args.tracker_name
tracker_param = args.tracker_param
dataset_name = args.dataset_name
runid = args.runid

trackers.extend(trackerlist(name=tracker_name, parameter_name=tracker_param, dataset_name=dataset_name, run_ids=runid, display_name=tracker_param))

# For VOT evaluate
dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True)

# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

