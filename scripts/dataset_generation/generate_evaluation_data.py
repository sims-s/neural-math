import numpy as np
from argparse import ArgumentParser
from sympy import factorint
from tqdm.auto import tqdm
import pandas as pd
from treelib import Node, Tree
from collections import defaultdict
import copy
import os

dict2obj = lambda d: type("Object", (), d)

tree_finished = lambda t: all([
                (node.is_leaf() and node.data.type == 'num') or 
                (not node.is_leaf() and node.data.type == 'expr') 
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


def sample_equation(distr, functions, vals):
    node_id = 0
    tree = Tree()
    tree.create_node(node_id, node_id, data = dict2obj({'type' : 'expr', 'val' : np.random.choice(functions)}))
    node_id += 1
    
    while not tree_finished(tree):
        
        run_dict = copy.deepcopy(tree.nodes)
        
        for node_name, node in run_dict.items():
            if node.is_leaf() and not node.data.type == 'num':
                assert node.data.type == 'expr'
                n_children = function_to_nargs[node.data.val]
                node_depth = tree.depth(node)
                
                for i in range(n_children):
                    type_ = np.random.choice(['expr', 'num'], p=distr[node_depth+1])
                    value = np.random.choice(functions if type_ == 'expr' else vals)
                    tree.create_node(node_id, node_id, data = dict2obj({'type' : type_, 'val' : value}), parent=node.identifier)
                    node_id += 1

    return tree

def convert_tree_to_equation(tree, full_parens=False):
    equation = ""

    def _traverse(node):
        nonlocal equation
        if node.is_leaf():
            equation += str(node.data.val)
        else:
            n_children = len(tree.children(node.identifier))
            for i, child in enumerate(tree.children(node.identifier)):
                if full_parens:
                    need_parens = child.data.type == 'expr'
                else:
                    need_parens = child.data.type == 'expr' and parent_needs_parens_for_child(node.data.val, child.data.val)
                    need_parens = need_parens and not (i==0 and order_operations[node.data.val] >= order_operations[child.data.val])
                
                if need_parens:
                    equation += "("
                _traverse(child)
                if need_parens:
                    equation += ")"
                if not i == n_children - 1:
                    equation += node.data.val

    _traverse(tree.nodes[tree.root])

    return equation


def parent_needs_parens_for_child(f1, f2):
    f1_idx = order_operations[f1]
    f2_idx = order_operations[f2]
    if not f1_idx == f2_idx:
        return f1 < f2
    else:
        return f1 in non_associative_functions

    
def create_sample_df(n_samples, probs, functions, values, oos_vals = None):
    if not n_samples:
        return None
    rows = []
    if oos_vals is not None:
        oos_vals = set(list(oos_vals))
    with tqdm(total = n_samples, leave=False) as pbar:
        while len(rows) < n_samples:
            # OOS equations awre ones that contain at least one number that is out of sample
            tree = sample_equation(probs, functions, values)
            if oos_vals is not None and not set([node.data.val for _, node in tree.nodes.items() if node.data.type=='num']).intersection(oos_vals):
                continue
            eqn = convert_tree_to_equation(tree)
            rows.append([eqn])
            pbar.update(1)
    return pd.DataFrame.from_records(rows, columns = ['equation'])



def main(args):
    exp_vs_num_prob = np.array([
        [.5, .5],
        [.5,.5],
        [0.,1.],
    ])
    functions = ['*', '+', '—']
    value_range_in_sample = np.arange(args.min_val, args.max_val)
    out_of_sample_vals = np.arange(args.max_val, int(args.max_val * 1.1))
    full_vals = np.arange(args.min_val, int(args.max_val * 1.1))

    keys = ['train', 'val', 'oos']
    dfs = {k : create_sample_df(getattr(args, f'{k}_samples'), exp_vs_num_prob, functions, full_vals if k=='oos' else value_range_in_sample, 
                                out_of_sample_vals if k=='oos' else None) for k in keys}
    
    os.makedirs(args.save_dir, exist_ok=True)

    for k, v in dfs.items():
        if v is not None:
            save_name = f'{k}{"_" + args.suffix if args.suffix else ""}.csv'
            v[['equation']].to_csv(args.save_dir + save_name, index=False)
            print(k, v.shape)

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--min_val', type=int, default=0, help='genreate integers from 2...max_val')
    parser.add_argument('--max_val', type=int, default=256, help='genreate integers from 2...max_val')
    parser.add_argument('--train_samples', type=int, default=100_000)
    parser.add_argument('--val_samples', type=int, default=10_000)
    parser.add_argument('--oos_samples', type=int, default=2048)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='data/evaluation/')
    parser.add_argument('--suffix', default='')


    args = parser.parse_args()
    main(args)