from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/public/datasets/got10k_lmdb'
    settings.got10k_path = '/public/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.gtot_dir = '/public/datasets/gtot'
    settings.itb_path = '/public/datasets/itb'
    settings.lasher_path = '/public/datasets/lasher'
    settings.lasher_test_dir = '/public/datasets/lasher/testingset'
    settings.lasot_extension_subset_path = '/public/datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/public/datasets/lasot_lmdb'
    settings.lasot_path = '/public/datasets/lasot'
    settings.network_path = '/public/workspace/xyl/ODTrack-RGBT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/public/datasets/nfs'
    settings.otb_path = '/public/datasets/otb'
    settings.prj_dir = '/public/workspace/xyl/ODTrack-RGBT'
    settings.result_plot_path = '/public/workspace/xyl/ODTrack-RGBT/output/test/result_plots'
    settings.results_path = '/public/workspace/xyl/ODTrack-RGBT/output/test/tracking_results'    # Where to store tracking results
    settings.rgbt210_dir = '/public/datasets/RGBT210'
    settings.rgbt234_dir = '/public/datasets/RGBT234'
    settings.save_dir = '/public/workspace/xyl/ODTrack-RGBT/output'
    settings.segmentation_path = '/public/workspace/xyl/ODTrack-RGBT/output/test/segmentation_results'
    settings.tc128_path = '/public/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/public/datasets/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/public/datasets/trackingnet'
    settings.uav_path = '/public/datasets/uav'
    settings.vot18_path = '/public/datasets/vot2018'
    settings.vot22_path = '/public/datasets/vot2022'
    settings.vot_path = '/public/datasets/VOT2019'
    settings.youtubevos_dir = ''

    settings.lasher_path = '/public/datasets/LasHeR'  # xyl 修改Lasher数据集的路径
    settings.vtuav_path = '/public/datasets/VTUAV'  # xyl 修改Lasher数据集的路径

    return settings

