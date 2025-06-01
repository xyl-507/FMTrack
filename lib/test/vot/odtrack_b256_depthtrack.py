import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.odtrack_class import run_vot_exp

run_vot_exp('odtrack', 'baseline_256_lasher_mamba_v18', vis=False, out_conf=True, channel_type='rgbd')

