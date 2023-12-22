import os
import time
import random
import math
import requests
import numpy as np
import pandas as pd
from copy import deepcopy
import numbers

from alpha_module import Alpha, AlphaStage



API_BASE = "https://api.worldquantbrain.com"

REGION = 'USA'
UNIVERSE = 'SECTOR_UTILITIES_TOP3000'
DECAY = 0
DELAY = 0
NEUTRALIZATION = 'SUBINDUSTRY' 

DATASET_ID = 'other84' #'model216'

POPULATION_SIZE = 100
GENERATION_EPOCH = 20
MUTATION_RATE = 0.25
OS_RATIO = 0.8
chromosome_len = 3



def check_session_timeout(s):
    try:
        result = s.get(f"{API_BASE}/authentication").json()["token"]["expiry"]
        return float(result)
    except:
        return 0.0

def start_session(): 

    lstatus = None
    cnt = 0
    stime = 0
    while lstatus is None or lstatus == requests.status_codes.codes.unauthorized or stime == 0:
        time.sleep(5)
        s = requests.Session()
        credential_email = os.environ.get('WQ_CREDENTIAL_EMAIL')
        credential_password = os.environ.get('WQ_CREDENTIAL_PASSWORD')
        s.auth = (credential_email, credential_password)
        r = s.post(f"{API_BASE}/authentication")

        lstatus = r.status_code
        stime = check_session_timeout(s)
        cnt += 1
        
    return s

def get_datafields(
        s,
        instrument_type: str = 'EQUITY',
        region: str = 'USA',
        delay: int = 1,
        universe: str = 'TOP3000',
        dataset_id: str = '',
        search: str = '',
        datafield_type = 'MATRIX'
    ):
        if len(search) == 0:
            url_template = f"{API_BASE}/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
                "&offset={x}" + f"&type={datafield_type}"
            count = s.get(url_template.format(x=0)).json()['count'] 
        else:
            url_template = f"{API_BASE}/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
                f"&search={search}" +\
                "&offset={x}" + f"&type={datafield_type}"
            count = 100
        
        datafields_list = []
        for x in range(0, count, 50):
            datafields = s.get(url_template.format(x=x))
            datafields_list.append(datafields.json()['results'])

        datafields_list_flat = [item for sublist in datafields_list for item in sublist]

        datafields_df = pd.DataFrame(datafields_list_flat)
        return datafields_df['id'].tolist()

worker_sess = start_session()
other455 = []
try:
    other455 = get_datafields(worker_sess, dataset_id='other455', region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='MATRIX')
except:
    other455 = []

pv13 = []
try:
    pv13 = get_datafields(worker_sess, dataset_id='pv13', region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='MATRIX')
except:
    pv13 = []
    
grp_data_lst = [] 
try:
    grp_data_lst = get_datafields(worker_sess, region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='GROUP')
except:
    grp_data_lst = []

data_lst = []
try:
    data_lst = get_datafields(worker_sess, dataset_id=f'{DATASET_ID}', region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='VECTOR')
except:
    data_lst = []
data_x_lst = data_lst # ['opt4_60_call_vola_delta45'] # filter(lambda x: 'call' in x and 'vola' in x and 'delta' in x, data_lst)
data_y_lst = data_lst # ['opt4_60_put_vola_delta45'] # filter(lambda x: 'put' in x and 'vola' in x and 'delta' in x, data_lst) #data_lst # ['close', 'eps' , 'cap', 'capex', 'equity', 'cash', 'cashflow', 'debt', 'debt_st', 'debt_lt', 'assets', 'adv20', 'volume']


x_lst = [ f"ts_backfill(vec_avg({d}), 252)" for d in data_x_lst ] # ['ts_backfill(vec_avg(oth84_1_wshactualeps), 132)'] # [ f"ts_backfill(({d}), 252)" for d in data_lst ] # ['ts_backfill(vwap, 252)'] # ['ts_backfill(vec_avg(oth84_1_wshactualeps), 132)']
y_lst = [ f"ts_backfill(vec_avg({d}), 252)" for d in data_y_lst ] # ['ts_backfill(vec_avg(oth84_1_lastearningseps), 132)'] # [ f"ts_backfill(({d}), 252)" for d in data_lst ] # ['ts_backfill(close, 252)'] # ['ts_backfill(vec_avg(oth84_1_lastearningseps), 132)']

day_lst = [3,4,5,7,10,15,22,44,66,132,198,252]
grp_lst =  [ f"densify({g})" for g in grp_data_lst+other455+pv13+['subindustry', 'industry', 'sector', 'exchange', 'country', 'market']] 


ops_x_map = {
# ts1op_map = 
   # 'when_grk1': 'trade_when(group_rank({x}, {g})>0.5, {x}, -1)',
   # 'when_grk0': 'trade_when(group_rank({x}, {g})<0.5, {x}, -1)',
   # 'when_trk1': 'trade_when(ts_rank({x}, {d})>0.5, {x}, -1)',
   # 'when_trk0': 'trade_when(ts_rank({x}, {d})<0.5, {x}, -1)',
   # 'when_std05': 'trade_when(ts_std_dev({x}, {d})>0.5, {x}, -1)',
   # 'when_std1': 'trade_when(ts_std_dev({x}, {d})>1, {x}, -1)',
   # 'when_std2': 'trade_when(ts_std_dev({x}, {d})>2, {x}, -1)',
   # 'when_std3': 'trade_when(ts_std_dev({x}, {d})>3, {x}, -1)',
    'ldv': 'last_diff_value({x}, {d})',
    'ts_delay': 'ts_delay({x}, {d})',
    'ts_rank': 'ts_rank({x}, {d})',
    'ts_quantile_gaussian': 'ts_quantile({x}, {d}, driver="gaussian")',
    'ts_quantile_uniform': 'ts_quantile({x}, {d}, driver="uniform")',
    'ts_quantile_cauchy': 'ts_quantile({x}, {d}, driver="cauchy")',
    'ts_frac': '{x}/ts_sum({x}, {d})',
    'ts_sum': 'ts_sum({x}, {d})',
    'ts_entropy': 'ts_entropy({x}, {d})',
    'ts_scale': 'ts_scale({x}, {d})',
    'ts_mean': 'ts_mean({x}, {d})',
    'ts_delta': 'ts_delta({x}, {d})',
    'ts_av_diff': 'ts_av_diff({x}, {d})',
    'ts_av_diff_': 'ts_av_diff({x}, {d})',
    'ts_av_diff__': 'ts_av_diff({x}, {d})',
    'ts_ir': 'ts_ir({x}, {d})',
    'ts_zscore': 'ts_zscore({x}, {d})',
    'ts_returns': "ts_returns({x}, {d})",
    'ts_std_dev': 'ts_std_dev({x}, {d})',
    'ts_skewness': 'ts_skewness({x}, {d})',
    'ts_kurtosis': 'ts_kurtosis({x}, {d})',
    'ts_max': "ts_max({x}, {d})",
    'ts_min': "ts_min({x}, {d})",
    'ts_max_diff': "ts_max_diff({x}, {d})",
    'ts_min_diff': "ts_min_diff({x}, {d})",
    'ts_median': "ts_median({x}, {d})",
    'ts_min_max_cps': "ts_min_max_cps({x}, {d}, f = 2)",
    'ts_min_max_diff': "ts_min_max_diff({x}, {d}, f = 0.5)",
    'ts_moment_0': "ts_moment({x}, {d}, k=0)",
    'ts_moment_1': "ts_moment({x}, {d}, k=1)",
    'ts_moment_2': "ts_moment({x}, {d}, k=2)",
    'ts_moment_3': "ts_moment({x}, {d}, k=3)",
    'ts_product': "ts_product({x}, {d})",
    'ts_regression_step': 'ts_regression({x}, ts_step({d}), {d})',
    'ts_regression_delay_1': 'ts_regression({x}, ts_delay({x}, 1), {d})',
    'ts_regression_delay': 'ts_regression({x}, ts_delay({x}, {d}), {d})',
    'ts_regression_grp_median': 'ts_regression({x}, group_median({x}, {g}), {d})',
    'ts_regression_grp_mean': 'ts_regression({x}, group_mean({x}, 1, {g}), {d})',
    'ts_corr_step': 'ts_corr({x}, ts_step({d}), {d})',
    'ts_corr_delay': 'ts_corr({x}, ts_delay({x}, {d}), {d})',
# }

# grp1op_map = {
    'grp_rank': 'group_rank({x}, {g})',
    'grp_rank_': 'group_rank({x}, {g})',
    'grp_rank__': 'group_rank({x}, {g})',
    'scale': 'scale({x}, scale=1, longscale=1, shortscale=1)',
    'grp_zscore': 'group_zscore({x}, {g})',
    'grp_neut': 'group_neutralize({x}, ({g}))',
    'grp_neut_': 'group_neutralize({x}, ({g}))',
    'grp_neut__': 'group_neutralize({x}, ({g}))',
    'grp_norm': 'group_normalize({x}, {g})',
    'grp_scale': 'group_scale({x}, {g})',
    'group_frac': '{x}/group_sum({x}, {g})',
    'grp_diff_median': '{x}-group_median({x}, {g})',
    'grp_diff_median_': '{x}-group_median({x}, {g})',
    'grp_diff_mean': '{x}-group_mean({x}, 1, {g})',
    'grp_diff_mean_adv': '{x}-group_mean({x}, cap, {g})',
    'grp_diff_mean_cap': '{x}-group_mean({x}, adv20, {g})',
    'grp_div_median': '{x}/group_median({x}, {g})',
    'grp_div_mean': '{x}/group_mean({x}, 1, {g})',
    'x': '({x})',
    'tanh': 'tanh({x})',
    'sigmoid': 'sigmoid({x})',
    'exp': 'exp({x})',
    'log': 'log({x})',
    'log_diff': 'log_diff({x})',
    's_log_1p': 's_log_1p({x})',
    'reverse': 'reverse({x})',
    'reverse_': 'reverse({x})',
    'reverse__': 'reverse({x})',
    'inverse': 'inverse({x})',
    'ts_mean': "ts_mean({x}, {d})",
    'ts_decay_linear': "ts_decay_linear({x}, {d})",
    'ts_decay_linear_': "ts_decay_linear({x}, {d})",
    'ts_decay_linear__': "ts_decay_linear({x}, {d})",
    'ts_decay_linear_av_diff': "ts_decay_linear(ts_av_diff({x}, {d})>0, {d})",
    'ts_decay_linear_delta_bin': "ts_decay_linear(ts_delta({x}, {d})>0, {d})",
    'ts_decay_linear_ts_rank_bin': "ts_decay_linear(ts_rank({x}, {d})>0.5, {d})",
    'ts_decay_linear_rank_bin': "ts_decay_linear(rank({x})>0.5, {d})",
    'ts_decay_exp_window': "ts_decay_exp_window({x}, {d})",
    'hump_1': "hump({x}, hump = 0.000001)",
    'hump_2': "hump({x}, hump = 0.00001)",
    'hump_3': "hump({x}, hump = 0.0001)",
    'winsorize_0': "winsorize({x}, std = 2)",
    'winsorize_1': "winsorize({x}, std = 3)",
    'winsorize_2': "winsorize({x}, std = 4)",
    'winsorize_3': "winsorize({x}, std = 5)",
    'winsorize_4': "winsorize({x}, std = 10)",
}

whenop_map = {
    'df_when_grk1': 'trade_when(group_rank({x}, {g})>0.5, {x}, -1)',
    'df_when_grk0': 'trade_when(group_rank({x}, {g})<0.5, {x}, -1)',
    'df_when_trk1': 'trade_when(ts_rank({x}, {d})>0.5, {x}, -1)',
    'df_when_trk0': 'trade_when(ts_rank({x}, {d})<0.5, {x}, -1)',
    'df_when_std05': 'trade_when(ts_std_dev({x}, {d})>0.5, {x}, -1)',
    'df_when_std1': 'trade_when(ts_std_dev({x}, {d})>1, {x}, -1)',
    'df_when_std2': 'trade_when(ts_std_dev({x}, {d})>2, {x}, -1)',
    'df_when_std3': 'trade_when(ts_std_dev({x}, {d})>3, {x}, -1)',
    
    'df_when_grk1_y': 'trade_when(group_rank({x}, {g})>0.5, {y}, -1)',
    'df_when_grk0_y': 'trade_when(group_rank({x}, {g})<0.5, {y}, -1)',
    'df_when_trk1_y': 'trade_when(ts_rank({x}, {d})>0.5, {y}, -1)',
    'df_when_trk0_y': 'trade_when(ts_rank({x}, {d})<0.5, {y}, -1)',
    'df_when_std05_y': 'trade_when(ts_std_dev({x}, {d})>0.5, {y}, -1)',
    'df_when_std1_y': 'trade_when(ts_std_dev({x}, {d})>1, {y}, -1)',
    'df_when_std2_y': 'trade_when(ts_std_dev({x}, {d})>2, {y}, -1)',
    'df_when_std3_y': 'trade_when(ts_std_dev({x}, {d})>3, {y}, -1)',
}


diff2op_map = {
    'sub': 'subtract({x}, {y})',
    'div': 'divide({x}, {y})',
    'mas': 'ts_mean({x}, {d})-ts_mean({y}, {d})',
    'mad': 'ts_mean({x}, {d})/ts_mean({y}, {d})',
    'mts': 'ts_returns({x}, {d})-ts_returns({y}, {d})',
    'vec_neut': 'vector_neut({x}, {y})',
    'ts_reg': 'ts_regression({x}, {y}, {d})',
    'ts_reg_grp_med': 'ts_regression({x}, group_median({y}, {g}), {d})',
    'ts_reg_grp_mean': 'ts_regression({x}, group_mean({y}, 1, {g}), {d})',
    'ts_reg_grp_mean_cap': 'ts_regression({x}, group_mean({y}, cap, {g}), {d})',
    'ts_reg_ret': 'ts_regression(ts_returns({x}), ts_returns({y}), {d})',
    'ts_vec_neut': 'ts_vector_neut({x}, {y}, {d})',
    'ts_vec_neut_grp_med': 'ts_vector_neut({x}, group_median({y}, {g}), {d})',
    'ts_vec_neut_grp_mean': 'ts_vector_neut({x}, group_mean({y}, 1, {g}), {d})',
    'ts_vec_neut_grp_mean_cap': 'ts_vector_neut({x}, group_mean({y}, cap, {g}), {d})',
    'ts_vec_neut_ret': 'ts_vector_neut(ts_returns({x}), ts_returns({y}), {d})',
    'ts_co_kurtosis': 'ts_co_kurtosis({x}, {y}, {d})',
    'ts_co_skewness': 'ts_co_skewness({x}, {y}, {d})',
    'ts_covariance': 'ts_covariance({x}, {y}, {d})',
    'regression_neut': 'regression_neut({x}, {y})',
    
    'sub_r': 'subtract({y}, {x})',
    'div_r': 'divide({y}, {x})',
    'mas_r': 'ts_mean({y}, {d})-ts_mean({x}, {d})',
    'mad_r': 'ts_mean({y}, {d})/ts_mean({x}, {d})',
    'mts_r': 'ts_returns({y}, {d})-ts_returns({x}, {d})',
    'vec_neut_r': 'vector_neut({y}, {x})',
    'ts_reg_r': 'ts_regression({y}, {x}, {d})',
    'ts_reg_grp_med_r': 'ts_regression({y}, group_median({x}, {g}), {d})',
    'ts_reg_grp_mean_r': 'ts_regression({y}, group_mean({x}, 1, {g}), {d})',
    'ts_reg_grp_mean_cap_r': 'ts_regression({y}, group_mean({x}, cap, {g}), {d})',
    'ts_reg_ret_r': 'ts_regression(ts_returns({y}), ts_returns({x}), {d})',
    'ts_vec_neut_r': 'ts_vector_neut({y}, {x}, {d})',
    'ts_vec_neut_grp_med_r': 'ts_vector_neut({y}, group_median({x}, {g}), {d})',
    'ts_vec_neut_grp_mean_r': 'ts_vector_neut({y}, group_mean({x}, 1, {g}), {d})',
    'ts_vec_neut_grp_mean_cap_r': 'ts_vector_neut({y}, group_mean({x}, cap, {g}), {d})',
    'ts_vec_neut_ret_r': 'ts_vector_neut(ts_returns({y}), ts_returns({x}), {d})',
    'ts_co_kurtosis_r': 'ts_co_kurtosis({y}, {x}, {d})',
    'ts_co_skewness_r': 'ts_co_skewness({y}, {x}, {d})',
    'ts_covariance_r': 'ts_covariance({y}, {x}, {d})',
    'regression_neut_r': 'regression_neut({y}, {x})',
}
ops_map = {}
ops_y_map = {}
diff2op2_map = {}
for k, v in ops_x_map.items():
    ops_y_map[f"{k}_y"] = v.replace('{x}', '{y}')
for k, v in diff2op_map.items():
    diff2op2_map[f"{k}_2"] = v # .replace('{x}', '{y}')
ops_map.update(ops_x_map)
ops_map.update(ops_y_map)
ops_map.update(diff2op_map)

print(f'Paramaters Approximate Space: {len(whenop_map) * pow(len(ops_map), chromosome_len-1) * len(x_lst) * len(y_lst) * len(day_lst) * len(grp_lst)}')
# ops_map.update(whenop_map)

# decay1op_map = {

# }

def passbyval(func):
    def new(*args):
        cargs = [deepcopy(arg) for arg in args]
        return func(*cargs)
    return new

@passbyval
def merge_trees(treeA, treeB):
    if not treeA and not treeB:
        return None
    if not treeA:
        return treeB
    if not treeB:
        return treeA
    if isinstance(treeA, OP) and isinstance(treeB, OP):
        treeA.op_type = merge_trees(treeA.op_type, treeB.op_type)
        treeA.x = merge_trees(treeA.x, treeB.x)
        treeA.d = merge_trees(treeA.d, treeB.d)
        treeA.g = merge_trees(treeA.g, treeB.g)
        treeA.y = merge_trees(treeA.y, treeB.y)
    else:
        treeA = random.choice([treeA, treeB])
    return treeA

class OP:
    def __init__(self, x_lst, y_lst, d_lst, g_lst, ops_map):
        self.x = None
        self.y = None
        self.d = None
        self.g = None
        self.op_type = None
        self.x_lst = x_lst
        self.y_lst = y_lst
        self.d_lst = d_lst
        self.g_lst = g_lst
        self.ops_map = ops_map
        self.rnd()
    def __repr__(self):
        eval_string = ''
        x = self.x
        y = self.y
        d = self.d
        g = self.g
        if self.op_type:
            eval_string = eval(f"f'{self.ops_map[self.op_type]}'")
        return f"{eval_string}"
    def rnd(self):
        if not self.op_type:
            self.op_type = random.choice(sorted(self.ops_map.keys()))
        if not self.x:
            self.x = random.choice(self.x_lst)
        if not self.y:
            self.y = random.choice(self.y_lst)
        if not self.d:
            self.d = random.choice(self.d_lst)
        if not self.g:
            self.g = random.choice(self.g_lst)
        return self
    

def roulette_wheel(population):
    n = len(population)
    shoot = random.randint(0, math.floor(math.pow(n,3)))
    select = min(math.floor(math.pow(shoot,1/3)), n-1)
    return population[select]

def sigmoid(x):
    epsilon = 1e-15
    return 1 / (1+np.exp(-x+epsilon))

def objective_scoring(raw_val, upper, lower, reverse = False):
    if reverse:
        v = -1 * ( raw_val - upper ) / (upper - lower)
        if v > 1:
            return (v-1)*0.5+1
        elif v < 0:
            return v*1.5
        else: 
            return v
        # val = 1 - raw_val/baseline
        # return #(sigmoid(val-2)-0.5) if val >= 0 else (sigmoid(val)-0.5)*3
    else:
        v = ( raw_val - lower ) / (upper - lower)
        if v > 1:
            return (v-1)*0.5+1
        elif v < 0:
            return v*1.5
        else: 
            return v
        # val = raw_val/baseline - 1
        # return v# (sigmoid(val-2)-0.5) if val >= 0 else (sigmoid(val)-0.5)*3


# class OpTree:
#     def __init__(self, depth, x_lst, y_lst, d_lst, g_lst, ops_map):
#         self.depth = depth
#         self.root = self.generate_tree(depth, x_lst, y_lst, d_lst, g_lst, ops_map)

#     def __repr__(self):
#         return self._repr_recursive(self.root)

#     def _repr_recursive(self, node):
#         if not node:
#             return ""

#         repr_string = repr(node)
#         # if isinstance(node, OP) and node.x:
#         #     repr_string += str(node)# f"{self._repr_recursive(node.x)}"

#         return repr_string
    
#     def generate_tree(self, depth, x_lst, y_lst, d_lst, g_lst, ops_map):
#         if depth == 0:
#             return OP(x_lst, y_lst, d_lst, g_lst, ops_map)

#         op_node = OP(x_lst, y_lst, d_lst, g_lst, ops_map)
#         op_node.x = self.generate_tree(depth - 1, x_lst, y_lst, d_lst, g_lst, ops_map)
#         op_node.y = self.generate_tree(depth - 1, x_lst, y_lst, d_lst, g_lst, ops_map)
#         return op_node
    

#     def get_nth_op_parent(self, n, current_node=None, parent=None, counter=None):
#         if counter is None:
#             # Initialize a counter for tracking the number of encountered OPs
#             counter = [0]

#         if current_node is None:
#             current_node = self.root

#         if not current_node:
#             return None, None

#         if isinstance(current_node, OP):
#             # Increment the counter when an OP is encountered
#             counter[0] += 1

#             # Return the parent and the OP if it's the N-th one
#             if counter[0] == n:
#                 return parent, current_node

#         # Continue traversing the tree
#         return self.get_nth_op_parent(n, current_node.x, current_node, counter)

#     def modify_nth_op(self, n, new_op1, new_op2):
#         parent, nth_op = self.get_nth_op_parent(n)

#         if parent and nth_op:
#             # Replace the N-th OP with the new OP
#             parent.x = new_op1
#             parent.y = new_op2
#         else:
#             print(f"There is no N-th OP in the tree.")
            

    # def get_nth_op(self, n):
    #     counter = [0]
    #     def traverse(node):
    #         if not node:
    #             return None
    #         if isinstance(node, OP):
    #             counter[0] += 1
    #             if counter[0] == n:
    #                 return node
    #         return traverse(node.x)
    #     return traverse(self.root)

def generate_tree(depth):
    if depth <= 1:
        return OP(x_lst, y_lst, day_lst, grp_lst, ops_map)
    
    when_val = ['ts_std_dev(returns, 5)', 'ts_std_dev(returns, 22)', 'ts_std_dev(returns, 66)', 'ts_std_dev(volume, 5)', 'ts_std_dev(volume, 22)', 'ts_corr(volume, ts_step(5), 5)', 'ts_corr(volume, ts_step(22), 22)', 'adv20', 'returns']

    if depth == chromosome_len:
        op_node = OP(when_val, y_lst, day_lst, grp_lst, whenop_map)
        op_node.y = generate_tree(depth - 1)
    else:
        op_node = OP(x_lst, y_lst, day_lst, grp_lst, ops_map)
        op_node.x = generate_tree(depth - 1)
        op_node.y = generate_tree(depth - 1)
    
    # parent_node = OP(], y_lst, day_lst, grp_lst, whenop_map)

    return op_node
    
def crossover(parent_a, parent_b):
    merged_tree = merge_trees(parent_a, parent_b)
    return merged_tree

def gen_expression():#, ts_ops_map=ts1op_map, grp_ops_map=grp1op_map, bin_ops_map=diff2op_map, decay_ops_map=decay1op_map, day_lst=day_lst, grp_lst=grp_lst):
    # return OP(x_lst=[ OP(x_lst=[ OP(x_lst=[ OP(x_lst=x_lst, y_lst=y_lst, ops_map=bin_ops_map, d_lst=day_lst, g_lst=grp_lst)], y_lst=y_lst, ops_map=grp_ops_map, d_lst=day_lst, g_lst=grp_lst)], y_lst=y_lst, ops_map=grp_ops_map, d_lst=day_lst, g_lst=grp_lst)], y_lst=y_lst, ops_map=decay_ops_map, d_lst=day_lst, g_lst=grp_lst)
    opt = generate_tree(chromosome_len) # OP( x_lst=[ OP(x_lst=[ OP(x_lst=[ OP(x_lst=[ OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map) ], y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map) ], y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map) ], y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map) ], y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map
    return opt
# return OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map)

def gen_population(size):
    population = []
    while len(population)<size:
        for i in range(size):
            exp = gen_expression()# OpTree(4, x_lst, y_lst, day_lst, grp_lst, ops_map)# gen_expression()
            population.append(exp)
        population = list(set(population))
    return population


def generate_alpha(
    regular: str,
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = "SUBINDUSTRY",
    delay: int = 1,
    decay: int = 0,
    truncation: float = 0.01,
    nan_handling: str = "OFF",
    unit_handling: str = "VERIFY",
    pasteurization: str = "ON",
    visualization: bool = False,
):
    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "nanHandling": nan_handling,
            "instrumentType": "EQUITY",
            "delay": delay,
            "universe": universe,
            "truncation": truncation,
            "unitHandling": unit_handling,
            "pasteurization": pasteurization,
            "region": region,
            "language": "FASTEXPR",
            "decay": decay,
            "neutralization": neutralization,
            "visualization": visualization,
        },
        "regular": regular,
    }
    return simulation_data

def annualized_sharpe(pnl_df):
    # trading_days_per_year = 252
    # annualized_sharpe_ratio = (pnl_df['Return'].groupby(pnl_df.index // trading_days_per_year).mean().divide(pnl_df['Return'].groupby(pnl_df.index // trading_days_per_year).std())).mean() * math.sqrt(trading_days_per_year)
    annualized_sharpe_lst = (pnl_df["Return"].resample("Y").mean() / pnl_df['Return'].resample("Y").std()) * math.sqrt(252)

    return annualized_sharpe_lst

def calculate_max_drawdown(cumulative_pnl):
    cumulative_max = cumulative_pnl.cummax()

    drawdown = cumulative_pnl - cumulative_max
    
    max_drawdown = -1*drawdown.min()

    return max_drawdown



def evolution(verbose=False):
    batch_size = POPULATION_SIZE
    parent_population = gen_population(batch_size)
    research_id_prefix = f'{REGION}_{UNIVERSE}_{DELAY}_{DATASET_ID}_{NEUTRALIZATION}'
    for e in range(GENERATION_EPOCH):
        alpha_batch = []
        print('\n')
        print('\t='*10)
        print(f"GENERATION {e}:")

        alpha_lst = [str(exp) for exp in parent_population ]
        for exp in alpha_lst:
            alpha = Alpha(name=f'{research_id_prefix}_{hash(exp)}', payload=generate_alpha(regular=exp, region=f'{REGION}', universe=f'{UNIVERSE}', delay=DELAY, decay=DECAY, neutralization=f'{NEUTRALIZATION}'))
            alpha.save_to_disk()
        while True:
            # TODO: validate research_id_prefix
            if len(os.listdir(AlphaStage.PENDING.value)) <=0 and len(os.listdir(AlphaStage.RUNNING.value)) <= 0:
                break
            else:
                time.sleep(5)
        complete_files = [os.path.join(AlphaStage.COMPLETED.value, file) for file in os.listdir(AlphaStage.COMPLETED.value)]
        for complete_file in complete_files:
            complete_alpha: Alpha = Alpha.load_from_disk(file_path=complete_file)

            pnl_df = pd.read_csv(complete_alpha.response_data['pnl_path'])
            pnl_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            pnl_df['Date'] = pd.to_datetime(pnl_df['Date'])
            pnl_df = pnl_df.set_index("Date")
            pnl_df = pd.DataFrame(pnl_df,columns=["Pnl"])
            pnl_df['Return'] = (pnl_df['Pnl'].copy().diff() / 10000000)

            tvr_df = pd.read_csv(complete_alpha.response_data['tvr_path'])
            tvr_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            tvr_df['Date'] = pd.to_datetime(tvr_df['Date'])
            tvr_df = tvr_df.set_index("Date")
            tvr_df = pd.DataFrame(tvr_df,columns=["Turnover"])

            
            pnl_df['Cumulative'] = pnl_df['Pnl'].round(2)
            pnl_df['HighValue'] = pnl_df['Cumulative'].cummax()
            pnl_df['Drawdown'] = (pnl_df['Cumulative'] - pnl_df['HighValue']) / pnl_df['HighValue']
            pnl_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            maxdrawdown_year = [ -1*dd for dd in (pnl_df['Drawdown'].resample("Y").mean()).values.tolist()]
            turnover_year = (tvr_df['Turnover'].resample("Y").mean()).values.tolist()
            # returns_year = np.multiply((pnl_df['Return'].resample("Y").sum()).values.tolist(), (252/pnl_df['Pnl'].resample("Y").count()).values.tolist())
            returns_year = (pnl_df['Return'].resample("Y").sum()).values.tolist()
            sharpe_year = ((pnl_df['Return'].resample("Y").mean() /  (pnl_df['Return'].resample("Y")).std())*math.sqrt(252)).values.tolist()
            margin_year = ((pnl_df['Pnl'].copy().diff() / (tvr_df['Turnover'] * (2*10000000))).resample("Y").mean()*10000).values.tolist()
            fitness_year = sharpe_year * np.sqrt(np.divide([ abs(ret) for ret in returns_year ], [ max(tvr, 0.125) for tvr in turnover_year])) 
            
            def is_valid_number(value):
                if isinstance(value, numbers.Number) and not isinstance(value, complex):
                    if value not in {float('inf'), float('-inf'), float('nan'), None}:
                        return True

                return False

            alpha_stats = complete_alpha.response_data
            if True: #is_valid_number(np.mean(turnover_year)) and is_valid_number(np.mean(sharpe_year)) and is_valid_number(np.mean(returns_year)) and is_valid_number(np.mean(maxdrawdown_year)) and is_valid_number(np.mean(margin_year)) and is_valid_number(np.mean(fitness_year)): 
                n = 1 # 6
                is_stats = {'sharpe': np.mean(sharpe_year[n:-2]), 'sharpe_lt':  np.mean(sharpe_year[n:-2]), 'sharpe_st':  np.mean(sharpe_year[-4:-2]), 'fitness': np.mean(fitness_year[n:-2]), 'turnover': np.mean(turnover_year[n:-2]), 'margin': np.mean(margin_year[n:-2]), 'drawdown': np.mean(maxdrawdown_year[n:-2]), 'returns': np.mean(returns_year[n:-2])} # alpha_stats['is']
                if float(is_stats['turnover'])>0 and float(is_stats['returns'])>0 and float(is_stats['sharpe'])>0 and float(is_stats['fitness'])>0: #float(is_stats['sharpe_st'])>0 and float(is_stats['sharpe_lt'])>0 and float(is_stats['turnover'])>0.01 and float(is_stats['turnover'])<1 and float(is_stats['drawdown']) < 0.5:
                    score = (objective_scoring(float(is_stats['sharpe_lt']), 3, 1.5) + objective_scoring(float(is_stats['sharpe_st']), 4, 2) + objective_scoring(float(is_stats['fitness']), 3, 1.5) + objective_scoring(max(float(is_stats['turnover']), 0.1), 0.6, 0.3, True) + objective_scoring(float(is_stats['margin']), 20, 10) + objective_scoring(float(is_stats['drawdown']), 0.05, 0.01, True))/6 # (objective_scoring(float(is_stats['fitness']), 1.5) + objective_scoring(float(is_stats['sharpe']), 1.6) + objective_scoring(float(is_stats['turnover']), 0.2, True) + objective_scoring(float(is_stats['returns']), 0.2) + objective_scoring(float(is_stats['drawdown']), 0.02, True) + objective_scoring(float(is_stats['margin']), 0.0015))/6

                    if is_valid_number(score) and (int(alpha_stats['is']['longCount'])+int(alpha_stats['is']['shortCount']))>500:# and np.mean(sharpe_year[-3:-1])>0:#abs(np.mean(sharpe_year[-2:-1])/np.mean(sharpe_year[-3:-1])) > 0.6:
                        for a_i in parent_population:
                            if str(a_i) == alpha_stats['regular']['code']:
                                alpha_batch.append({'id': alpha_stats['id'], 'score': score, 'data': a_i, 'fitness':is_stats['fitness'], 'sharpe':is_stats['sharpe'], 'margin': is_stats['margin'], 'turnover':is_stats['turnover'], 'drawdown': is_stats['drawdown'], 'returns': is_stats['returns']}) # , 'fitness': is_stats['fitness'], 'returns': is_stats['returns'], 'drawdown': is_stats['drawdown'], 'margin': is_stats['margin']
                                break

        alpha_rank_batch = sorted(alpha_batch, key=lambda x: x['score'], reverse=False)
        
        for v in alpha_rank_batch:
            # print(f"https://platform.worldquantbrain.com/alpha/{v['id']} :\t{round(v['score'], 2)}\t{v['fitness']}\t{v['sharpe']}\t{round(v['turnover']*100,2)}\t{round(v['returns']*100,2)}\t{round(v['drawdown']*100,2)}\t{round(v['margin']*10000,2)}") #\t{v['corr']>0.995}")
            print(f"https://platform.worldquantbrain.com/alpha/{v['id']} : Fitness: {round(v['fitness'], 2)} Sharpe: {round(v['sharpe'], 2)} Turnover: {round(v['turnover']*100,2)} Returns: {round((v['returns'])*100,2)} Turnover: {round((v['turnover'])*100,2)} Drawdown: {round((v['drawdown']),2)} Score: {round(v['score'],2)}") #\t{v['corr']>0.995}")

        children_population = []
        batch_size /= 1.05
        batch_size = int(batch_size)
        while len(children_population) < batch_size:
            parent_a , parent_b = roulette_wheel([x['data'] for x in alpha_rank_batch]), roulette_wheel([x['data'] for x in alpha_rank_batch])
            child = crossover(parent_a, parent_b)
            if random.random() < MUTATION_RATE:
                # rn = random.randint(int(child.depth/2), child.depth)
                # child.modify_nth_op(child.depth, OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map), OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map))
                child = generate_tree(chromosome_len)
                    #child.x.x.x.x = OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map)# gen_expression()
                    #child.y.y.y.y = OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map)# gen_expression()
                # else:
                    #child.x.x.x = OP(x_lst= [ OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map) ] , y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map)
                    #child.y.y.y = OP(x_lst= [ OP(x_lst=x_lst, y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map) ] , y_lst=y_lst, d_lst=day_lst, g_lst=grp_lst, ops_map=ops_map)
            children_population.append(child)
        
        parent_population = children_population


evolution()
