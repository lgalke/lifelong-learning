#!/usr/bin/bash 
python3 analysis/compute_tdiff_dist.py --max-hops 1 --save tdiff/12v-1hop data/dblp-easy
python3 analysis/compute_tdiff_dist.py --max-hops 2 --save tdiff/12v-2hop data/dblp-easy
python3 analysis/compute_tdiff_dist.py --max-hops 3 --save tdiff/12v-3hop data/dblp-easy
python3 analysis/compute_tdiff_dist.py --max-hops 1 --save tdiff/hard-1hop data/dblp-hard
python3 analysis/compute_tdiff_dist.py --max-hops 2 --save tdiff/hard-2hop data/dblp-hard
python3 analysis/compute_tdiff_dist.py --max-hops 3 --save tdiff/hard-3hop data/dblp-hard
python3 analysis/compute_tdiff_dist.py --max-hops 1 --save tdiff/pharma-1hop data/pharmabio
python3 analysis/compute_tdiff_dist.py --max-hops 2 --save tdiff/pharma-2hop data/pharmabio
python3 analysis/compute_tdiff_dist.py --max-hops 3 --save tdiff/pharma-3hop data/pharmabio
