import os
import time
import random
import math
import requests
import numpy as np
import pandas as pd
from copy import deepcopy

from alpha_module import Alpha, AlphaStage



API_BASE = "https://api.worldquantbrain.com"

REGION = 'TWN'
UNIVERSE = 'TOP500'
DECAY = 0
DELAY = 1
NEUTRALIZATION = 'SUBINDUSTRY'

DATASET_ID = 'other176'

POPULATION_SIZE = 200
GENERATION_EPOCH = 30
MUTATION_RATE = 0.3



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

# other455 = get_datafields(worker_sess, dataset_id='other455', region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='MATRIX')
# pv30 = get_datafields(worker_sess, dataset_id='pv30', region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='MATRIX')
grp_data_lst = get_datafields(worker_sess, region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='GROUP')

data_lst = get_datafields(worker_sess, dataset_id=f'{DATASET_ID}', region=f'{REGION}', delay=DELAY, universe=f'{UNIVERSE}', datafield_type='MATRIX')


x_lst = [ f"ts_backfill(({d}), 22)" for d in data_lst ] # vec_avg()

day_lst = [2,3,4,5,10,22,44,66,132,252]
grp_lst =  [ f"densify(group_coalesce({g}, subindustry))" for g in grp_data_lst ] # +other455+pv30 # ['subindustry', 'industry', 'sector', 'market', 'exchange', 'country'] + 

ts1op_map = {
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
    'ts_zscore': 'ts_zscore({x}, {d})',
    'ts_returns': "ts_returns({x}, {d})",
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
}

grp1op_map = {
    'grp_rank': 'group_rank({x}, {g})',
    'scale': 'scale({x}, scale=1, longscale=1, shortscale=1)',
    'grp_zscore': 'group_zscore({x}, {g})',
    'grp_neut': 'group_neutralize({x}, ({g}))',
    'grp_norm': 'group_normalize({x}, {g})',
    'grp_scale': 'group_scale({x}, {g})',
    'group_frac': '{x}/group_sum({x}, {g})',
    'grp_diff_median': '{x}-group_median({x}, {g})',
    'grp_diff_mean': '{x}-group_mean({x}, 1, {g})',
    'grp_div_median': '{x}/group_median({x}, {g})',
    'grp_div_mean': '{x}/group_mean({x}, 1, {g})',
}


decay1op_map = {
    'ts_mean': "ts_mean({x}, {d})",
    'ts_decay_linear': "ts_decay_linear({x}, {d})",
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
        # treeA.y = merge_trees(treeA.y, treeB.y)
    else:
        treeA = random.choice([treeA, treeB])
    return treeA

class OP:
    def __init__(self, x_lst, y_lst, d_lst, g_lst, ops_map):
        self.x = None
        # self.y = None
        self.d = None
        self.g = None
        self.op_type = None
        self.x_lst = x_lst
        # self.y_lst = y_lst
        self.d_lst = d_lst
        self.g_lst = g_lst
        self.ops_map = ops_map
        self.rnd()
    def __repr__(self):
        eval_string = ''
        x = self.x
        # y = self.y
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
        # if not self.y:
        #     self.y = random.choice(self.y_lst)
        if not self.d:
            self.d = random.choice(self.d_lst)
        if not self.g:
            self.g = random.choice(self.g_lst)
        return self
    

def roulette_wheel(population):
    n = len(population)
    shoot = random.randint(0, math.floor(n*n))
    select = min(math.floor(math.pow(shoot,1/2)), n-1)
    return population[select]

def sigmoid(x):
    return 1 / (1+math.exp(-x))

def objective_scoring(raw_val, baseline, reverse = False):
    if reverse:
        val = 1 - raw_val/baseline
        return sigmoid(val) if val >= 0 else val*2
    else:
        val = raw_val/baseline - 1
        return sigmoid(val) if val >= 0 else val*2
    
def crossover(parent_a, parent_b):
    merged_tree = merge_trees(parent_a, parent_b)
    return merged_tree

def gen_expression(x_lst=[], y_lst=[], ts_ops_map=ts1op_map, grp_ops_map=grp1op_map, decay_ops_map=decay1op_map, day_lst=day_lst, grp_lst=grp_lst):
    return OP(x_lst=[OP(x_lst=[ OP(x_lst=[ OP(x_lst=x_lst, y_lst=y_lst, ops_map=ts_ops_map, d_lst=day_lst, g_lst=grp_lst)], y_lst=y_lst, ops_map=grp_ops_map, d_lst=day_lst, g_lst=grp_lst)], y_lst=y_lst, ops_map=decay_ops_map, d_lst=day_lst, g_lst=grp_lst)], y_lst=y_lst, ops_map=grp_ops_map, d_lst=day_lst, g_lst=grp_lst)

def gen_population(size):
    population = []
    while len(population)<size:
        for i in range(size):
            exp = gen_expression(x_lst=x_lst)
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
    trading_days_per_year = 252
    annualized_sharpe_ratio = (pnl_df['Return'].groupby(pnl_df.index // trading_days_per_year).mean().divide(pnl_df['Return'].groupby(pnl_df.index // trading_days_per_year).std())).mean() * math.sqrt(trading_days_per_year)

    return annualized_sharpe_ratio

    

def evolution(verbose=False):
    
    parent_population = gen_population(POPULATION_SIZE)
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
            tvr_df = pd.read_csv(complete_alpha.response_data['tvr_path'])

            # is_cutoff = '2019-01-01'
            self_is_pnl, self_os_pnl = pnl_df.iloc[:int(len(pnl_df)*0.8),:], pnl_df.iloc[int(len(pnl_df)*0.8):,:] #pnl_df.loc[pnl_df.index < is_cutoff], pnl_df.loc[pnl_df.index >= is_cutoff]
            self_is_tvr, self_os_tvr = tvr_df.iloc[:int(len(tvr_df)*0.8),:], tvr_df.iloc[int(len(tvr_df)*0.8):,:] #tvr_df.loc[tvr_df.index < is_cutoff], tvr_df.loc[tvr_df.index >= is_cutoff]
            
            
            self_is_pnl['Return'] = self_is_pnl['Pnl'].diff() / 20000000

            self_is_pnl.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            annualized_sharpe_ratio = annualized_sharpe(self_is_pnl)
            
            average_daily_turnover = self_is_tvr['Turnover'].mean()

            alpha_stats = complete_alpha.response_data
            if annualized_sharpe_ratio > 0 and average_daily_turnover > 0:
                is_stats = {'sharpe': annualized_sharpe_ratio, 'turnover': average_daily_turnover} # alpha_stats['is']

                if is_stats['sharpe']:
                    score = (objective_scoring(float(is_stats['sharpe']), 1.6) + objective_scoring(max(float(is_stats['turnover']), 0.125), 0.2, True)) # (objective_scoring(float(is_stats['fitness']), 1.5) + objective_scoring(float(is_stats['sharpe']), 1.6) + objective_scoring(float(is_stats['turnover']), 0.2, True) + objective_scoring(float(is_stats['returns']), 0.2) + objective_scoring(float(is_stats['drawdown']), 0.02, True) + objective_scoring(float(is_stats['margin']), 0.0015))/6
                else:
                    score = -9999

                for a_i in parent_population:
                    if str(a_i) == alpha_stats['regular']['code'] and score != -9999:
                        alpha_batch.append({'id': alpha_stats['id'], 'score': score, 'data': a_i, 'sharpe':is_stats['sharpe'], 'turnover':is_stats['turnover']}) # , 'fitness': is_stats['fitness'], 'returns': is_stats['returns'], 'drawdown': is_stats['drawdown'], 'margin': is_stats['margin']
                        break

        alpha_rank_batch = sorted(alpha_batch, key=lambda x: x['score'], reverse=False)
        
        for v in alpha_rank_batch:
            # print(f"https://platform.worldquantbrain.com/alpha/{v['id']} :\t{round(v['score'], 2)}\t{v['fitness']}\t{v['sharpe']}\t{round(v['turnover']*100,2)}\t{round(v['returns']*100,2)}\t{round(v['drawdown']*100,2)}\t{round(v['margin']*10000,2)}") #\t{v['corr']>0.995}")
            print(f"https://platform.worldquantbrain.com/alpha/{v['id']} :\t{round(v['score'], 2)}\t{round(v['sharpe'], 2)}\t{round(v['turnover']*100,2)}") #\t{v['corr']>0.995}")

        children_population = []
        # POPULATION_SIZE -= 5
        while len(children_population) < POPULATION_SIZE:
            parent_a , parent_b = roulette_wheel([x['data'] for x in alpha_rank_batch]), roulette_wheel([x['data'] for x in alpha_rank_batch])
            child = crossover(parent_a, parent_b)
            if random.random() < MUTATION_RATE:
                child  = gen_expression(x_lst=x_lst)
            children_population.append(child)
        
        parent_population = children_population


evolution()
