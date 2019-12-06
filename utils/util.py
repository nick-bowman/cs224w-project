import os
from datetime import datetime
import time

home = os.path.expanduser("~")
base = os.path.join(home, "WikiLinksGraph/WikiLinksGraph")
project_base = os.path.join(home, "cs224w-project")

def generate_file_name(language, year):
    file_name = "%swiki.wikilink_graph.%s-03-01.csv" % (language, str(year))
    return os.path.join(base, file_name)

def generate_diff_file_name(lang, curr, future):
    file_name = f"{lang}wiki-{curr}-{future}-diff.csv"
    return os.path.join(project_base, "data_diffs", file_name)

def load_diff(language, curr_year, future_year):
    all_edges = set()
    with open(generate_diff_file_name(language, curr_year, future_year), "r") as f:
        for line in f: 
            src, dst = line.split()
            all_edges.add((int(src),int(dst)))
    return all_edges

def evaluate_predicted_edges(lang, curr_year, future_year, predicted_edges, filter_nodes=None):
    true_added_edges = load_diff(lang, curr_year, future_year)
#     print(predicted_edges)
#     print(true_added_edges)
    
    to_remove = set()
    if filter_nodes:
        for e in true_added_edges:
            if e[0] not in filter_nodes or e[1] not in filter_nodes:
                to_remove.add(e)
    true_added_edges -= to_remove

    false_positive = len(predicted_edges - true_added_edges)
    false_negative = len(true_added_edges - predicted_edges)
    true_positive = len(true_added_edges.intersection(predicted_edges))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)

def timestamp_to_int(timestamp):
    """
    Takes a string timestamp in the form
    
        year-month-day hour-minute-second timezone
        
    and converts it into an integer.
    
    Parameters
    ----------
    timestamp : str
        Timestamp in the form specified above.
        
    Returns
    -------
    int_timestamp : int
        Timestamp in integer form.
    """
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S %Z")
    return int(time.mktime(dt.timetuple()))