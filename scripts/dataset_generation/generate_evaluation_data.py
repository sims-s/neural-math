import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd
from treelib import Node, Tree
from collections import defaultdict
import evaluation_configs
import os
import json

# Randomly sample everything we need to generate expression  in batch
# calling np.random.choice(...) 1000 times is slower than np.random.choice(..., size=1000)
class ExpressionSampleCache():
    def __init__(self, size, functions, vals, distr):
        self.size = size
        self.functions = functions
        self.vals = vals
        self.distr = distr
        self.data = {
            'functions' : {
                'counter' : 0,
                'values' : np.random.choice(functions, size)
            },
            'vals' : {
                'counter' : 0,
                'values' : np.random.choice(vals, size)
            },
            'types' : {
                    i+1 : {
                        'counter' : 0,
                        'values' : np.random.choice(['expr', 'num'], size=size, p=d)
                    }
                    for i, d in enumerate(distr[1:])
            },
        }
    
    def restart(self, key, depth=-1):
        if key == 'functions':
            self.data['functions'] = {
                'counter' : 0,
                'values' : np.random.choice(self.functions, self.size)
            }
        elif key == 'vals' : 
            self.data['vals'] = {
                'counter' : 0,
                'values' : np.random.choice(self.vals, self.size)
            }
        elif key == 'types' :
            assert depth > 0, 'need to specify depth for type'
            self.data['types'][depth] = {
                        'counter' : 0,
                        'values' : np.random.choice(['expr', 'num'], size=self.size, p=self.distr[depth])
                    }
    
    def sample(self, sample_type, depth=-1):
        subdict = self.data[sample_type]
        if sample_type == 'types':
            assert depth > 0, 'need to specify depth for type'
            subdict = subdict[depth]
        if subdict['counter'] >= len(subdict['values']):
            self.restart(sample_type, depth)
            subdict = self.data[sample_type]
            if sample_type == 'types':
                subdict = subdict[depth]
        
        to_return = subdict['values'][subdict['counter']]
        subdict['counter'] += 1
        
        return to_return


# Tree & Node used for sampling equations           
class Tree():
    def __init__(self):
        self.nodes = {}
        self.depth = {}
        self.root = None
        
    def create_node(self, id_, data, parent=None):
        node = Node(id_, data, parent)
        self.nodes[id_] = node
        if parent is not None:
            self.depth[id_] = self.depth[parent] + 1
            self.nodes[parent].children.append(node)
        else:
            self.depth[id_] = 0
            assert self.root is None, "only root has no parents"
            self.root = id_
        
    def root(self):
        return self.root
        
    def depth(self, node_id):
        return self.depth[node_id]
    
    def children(self, node_id):
        tgt = self.nodes[node_id]
        for child in tgt.children:
            yield child
        
    
    def __repr__(self):
        return str(self.nodes)
               
class Node():
    def __init__(self, id_, data, parent=None):
        self.id = id_
        self.data = data
        self.parent = parent
        self.children = []
        
    def is_leaf(self):
        return len(self.children) == 0 
    
    def __repr__(self):
        return str(self.data)




def sample_expression(cache):
    node_id = 0
    tree = Tree()
    if not cache.distr[0,0] == 1:
        raise ValueError('should sample a function first')
    tree.create_node(node_id, data = {'type' : 'expr', 'val' : cache.sample('functions')})
    node_id += 1
    
    while not tree_finished(tree):
        run_dict = {k:v for k, v in tree.nodes.items() if v.is_leaf() and not v.data['type']=='num'}

        for node_name, node in run_dict.items():
            n_children = function_to_nargs[node.data['val']]
            node_depth = tree.depth[node_name]
            for i in range(n_children):
                type_ = cache.sample('types', depth=node_depth+1)
                value = cache.sample('vals' if type_=='num' else 'functions')

                tree.create_node(node_id, data = {'type' : type_, 'val' : value}, parent=node_name)
                node_id += 1
    
    return tree

def convert_tree_to_expression(tree, full_parens=False):
    expression = ""

    def _traverse(node):
        nonlocal expression
        if node.is_leaf():
            expression += str(node.data['val'])
        else:
            n_children = len(node.children)
            
            for i, child in enumerate(node.children):
                if full_parens:
                    need_parens = child.data['type'] == 'expr'
                else:
                    need_parens = child.data['type'] == 'expr' and parent_needs_parens_for_child(node.data['val'], child.data['val'])
                    need_parens = need_parens and not (i==0 and order_operations[node.data['val']] >= order_operations[child.data['val']])
                
                if need_parens:
                    expression += "("
                _traverse(child)
                if need_parens:
                    expression += ")"
                if not i == n_children - 1:
                    expression += node.data['val']
    
    _traverse(tree.nodes[tree.root])

    return expression


def parent_needs_parens_for_child(f1, f2):
    f1_idx = order_operations[f1]
    f2_idx = order_operations[f2]
    if not f1_idx == f2_idx:
        return f1 < f2
    else:
        return f1 in non_associative_functions

    
def create_sample_df(n_samples, probs, functions, values, oos_vals = None, sample_cache_size=50_000, invalid_exprs=None):
    if not n_samples:
        return None
    if invalid_exprs is None:
        invalid_exprs = set()

    sampler_cache = ExpressionSampleCache(sample_cache_size, functions, values, probs)
    rows = set()
    if oos_vals is not None:
        oos_vals = set(list(oos_vals))
    with tqdm(total = n_samples, leave=False) as pbar:
        while len(rows) < n_samples:
            # OOS expressions awre ones that contain at least one number that is out of sample
            tree = sample_expression(sampler_cache)
            if oos_vals is not None and not set([node.data['val'] for _, node in tree.nodes.items() if node.data['type']=='num']).intersection(oos_vals):
                continue
            expr = convert_tree_to_expression(tree)
            if expr in rows or expr in invalid_exprs:
                continue
            rows.add(expr)
            pbar.update(1)

    return pd.Series(list(rows), name='expression').to_frame()#pd.DataFrame.from_records(list(rows), columns = ['expression'])


tree_finished = lambda t: all([
                (not node.is_leaf() and node.data['type'] == 'expr') or
                (node.is_leaf() and node.data['type'] == 'num')
                for _, node in t.nodes.items()])

order_operations = {
    '^' : 0,
    '*' : 1,
    '/' : 1,
    '—' : 2,
    '+' : 2
}
non_associative_functions = ['^', '/', '—']
function_to_nargs = defaultdict(lambda: 2)


def get_n_samples(args, config):
    results = {}
    keys = ['train', 'val', 'oos']
    for k in keys:
        k = f'{k}_samples'
        from_config = -1 if not k in config else config[k]
        from_args = getattr(args, k)
        if from_args >= 0:
            results[k] = from_args
        elif from_config >= 0:
            results[k] = from_config
    return results['train_samples'], results['val_samples'], results['oos_samples']
        


def main(args):
    np.random.seed(args.seed)
    
    config = evaluation_configs.CONFIGS[args.config]
    for k in ['exp_vs_num_prob', 'in_sample_vals', 'oos_vals']:
        if not isinstance(config[k], np.ndarray):
            config[k] = np.array(config[k])

    
    exp_vs_num_prob = config['exp_vs_num_prob']
    functions = config['functions']
    

    value_range_in_sample = config['in_sample_vals']
    out_of_sample_vals = config['oos_vals']
    full_vals = config['full_vals']

    train_samples, val_samples, oos_samples = get_n_samples(args, config)

    dfs = {}
    dfs['val'] = create_sample_df(val_samples, exp_vs_num_prob, functions, value_range_in_sample, None)
    # print('Made VAL!')
    dfs['oos'] = create_sample_df(oos_samples, exp_vs_num_prob, functions, full_vals, out_of_sample_vals)
    # print('Made OOS!')

    val_vals = None if dfs['val'] is None else set(dfs['val']['expression'].tolist())
    dfs['train'] = create_sample_df(train_samples, exp_vs_num_prob, functions, value_range_in_sample, None, invalid_exprs=val_vals)
    # print('Made Train!')

    
    os.makedirs(args.save_dir, exist_ok=True)

    for k, v in dfs.items():
        if v is not None:
            save_name = f'{k}{"_" + args.suffix if args.suffix else "_" + args.config}.csv'
            v[['expression']].to_csv(args.save_dir + save_name, index=False)
            print(k, v.shape)

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    parser.add_argument('--train_samples', type=int, default=-1)
    parser.add_argument('--val_samples', type=int, default=-1)
    parser.add_argument('--oos_samples', type=int, default=-1)
    parser.add_argument('--sample_cache_size', type=int, default=50_000)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='data/evaluation/')
    parser.add_argument('--suffix', default='')


    args = parser.parse_args()
    main(args)