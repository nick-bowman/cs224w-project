import networkx as nx
from utils.util import project_base, generate_file_name, generate_gold_file_name, load_networkx_graph, get_num_lines
import os
import shutil
from random import choice 
from tqdm import tqdm 

def create_dataset(args):
    graph_filename = generate_file_name(args.lang, args.year)
    g = load_networkx_graph(graph_filename)
    gold_filename = generate_gold_file_name(args.lang, args.year, args.year + 1, args.k)
    gold_g = load_networkx_graph(gold_filename, src_index=0, dst_index=1)
    
    nodes = g.nodes()
    node_list = list(nodes)
    edges = g.edges()
    gold_edges = g.edges()
    
    
    # generate random ordering of gold edges
    shuffle_command = f"shuf {gold_filename} -o {gold_filename}"
    os.system(shuffle_command)
    
    num_gold_edges = get_num_lines(gold_filename)
    num_edges_in_dataset = int(num_gold_edges / args.positive_frac)
    num_negative_edges = num_edges_in_dataset - num_gold_edges
        
    base_folder = os.path.join("datasets", f"{args.lang}wiki-{args.year}-{args.k}")
    if os.path.exists(base_folder) and os.path.isdir(base_folder):
        shutil.rmtree(base_folder)
    os.mkdir(base_folder)
    
    training_data_file = os.path.join(base_folder, "training.csv")
    testing_data_file = os.path.join(base_folder, "test.csv")
    validation_data_file = os.path.join(base_folder, "validation.csv")
    
    positive_edges_file = os.path.join(base_folder, "all_positive_edges.csv")
    negative_edges_file = os.path.join(base_folder, "all_negative_edges.csv")
    negative_edges_added = 0
    pbar = tqdm(total = num_negative_edges)
    with open(negative_edges_file, "w") as f: 
        while negative_edges_added < num_negative_edges: 
            src = choice(node_list)
            dst = choice(node_list)
            edge = (src,dst)
            if edge not in edges and edge not in gold_edges:
                f.write(f"{src}\t{dst}\n")
                negative_edges_added += 1
                pbar.update(1)
        pbar.close()
    
    
    shutil.copyfile(gold_filename, positive_edges_file)
    
    os.system(f"sed -i 's/$/\t1/' {positive_edges_file}")
    os.system(f"sed -i 's/$/\t0/' {negative_edges_file}")
    
    num_positive = num_gold_edges
    num_negative = num_negative_edges
    
    num_positive_train = int(args.train * num_positive)
    num_positive_test = int(args.test * num_positive)
    num_positive_val = num_positive - num_positive_train - num_positive_test
        
    num_negative_train = int(args.train * num_negative)
    num_negative_test = int(args.test * num_negative)
    num_negative_val = num_negative - num_negative_train - num_negative_test 
    
    os.system(f"sed -e '{num_positive_train}q' {positive_edges_file} > {training_data_file}")
    os.system(f"sed -e '1,{num_positive_train}d;{num_positive_train + num_positive_test}q' {positive_edges_file} > {testing_data_file}")
    os.system(f"sed -e '1,{num_positive_train + num_positive_test}d' {positive_edges_file} > {validation_data_file}")
    
    os.system(f"sed -e '{num_negative_train}q' {negative_edges_file} >> {training_data_file}")
    os.system(f"sed -e '1,{num_negative_train}d;{num_negative_train + num_negative_test}q' {negative_edges_file} >> {testing_data_file}")
    os.system(f"sed -e '1,{num_negative_train + num_negative_test}d' {negative_edges_file} >> {validation_data_file}")
    
    shuffle_command = f"shuf {training_data_file} -o {training_data_file}"
    os.system(shuffle_command)
    
    shuffle_command = f"shuf {testing_data_file} -o {testing_data_file}"
    os.system(shuffle_command)
    
    shuffle_command = f"shuf {validation_data_file} -o {validation_data_file}"
    os.system(shuffle_command)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lang', default="en", type=str, 
        help='Language for which to create dataset',
    )
    parser.add_argument(
        '--year', default=2001, type=int, 
        help='The year of which snapshot to use',
    )
    parser.add_argument(
        '--train', default=0.8, type=float, 
        help='Percentage of the graph to use for training',
    )
    parser.add_argument(
        '--test', default=0.1, type=float, 
        help='Percentage of the graph to use for testing',
    )
    parser.add_argument(
        '--val', default=0.1, type=float, 
        help='Percentage of the graph to use for validation',
    )
    parser.add_argument(
        '-k', default=3, type=int,
        help="k-value to use for gold edge lookup"
    )
    parser.add_argument(
        '--positive_frac', default=0.1, type=float,
        help="percentage of positive edges in the dataset"
    )

    args = parser.parse_args()

    create_dataset(args)