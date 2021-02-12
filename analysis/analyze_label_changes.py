import os.path as osp
import json
import argparse
import numpy as np
import pandas as pd

def resolve_classes(class_set, index2class):
    return [index2class[c] for c in list(class_set)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to graph dataset")
    parser.add_argument('--start', type=int, help="Start year")
    parser.add_argument('--end', type=int, help="Start year")

    args = parser.parse_args()
    data = pd.DataFrame(
        {
          'year': np.load(osp.join(args.path, 't.npy')),
         'label': np.load(osp.join(args.path, 'y.npy'))
        }
    )
    t_start = args.start if args.start is not None else data.year.min()
    t_end = args.end if args.end is not None else data.year.max()

    with open(osp.join(args.path, 'label2index.json'), 'r') as fh:
        label2index = json.load(fh)

    index2label = {v: k for k, v in label2index.items()}
    print(label2index.keys())
    
    classes = set(data[data.year == t_start].label)
    print("~ Year:", t_start, '~')
    print("#cls:", len(classes))
    n_add_total = 0
    n_rem_total = 0
    classes_total = classes
    for t in range(t_start+1, t_end+1):
        print("~ Year:", t, '~')
        next_classes = set(data[data.year == t].label)
        add = next_classes - classes
        rem = classes - next_classes
        print("add:", " | ".join(resolve_classes(add, index2label)))
        print("rem:", " | ".join(resolve_classes(rem, index2label)))
        n_add = len(add)
        n_rem = len(rem)
        print("#cls:", len(next_classes))
        print("#add:", n_add)
        print("#rem:", n_rem)
        n_add_total += n_add
        n_rem_total += n_rem
        classes = next_classes
        classes_total |= next_classes


    print("~~~ Total ~~~")
    print("#add:", n_add_total)
    print("#rem:", n_rem_total)
    print("#diff:", n_add_total + n_rem_total)
    print("#cls:", len(classes_total))


if __name__ == "__main__":
    main()
