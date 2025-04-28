#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09/04/2025 11:26
# @Author  : mmai
# @FileName: run_experiments
# @Software: PyCharm

import argparse
import json
import os
import time
import pandas as pd
from mable.examples import environment, fleets, companies
import greedy
import kbest_bid

def parse_args():
    parser = argparse.ArgumentParser(description='Run shipping experiments with KBestBid')
    
    # Experiment settings
    parser.add_argument('--mode', type=str, choices=['single', 'sweep', 'subset'], default='subset',
                      help='Run a single experiment, full parameter sweep, or subset of combinations')
    parser.add_argument('--output', type=str, default='results.csv',
                      help='Output file for results')
    parser.add_argument('--config', type=str, default=None,
                      help='JSON config file for parameter sweep')
    parser.add_argument('--subset-size', type=int, default=10,
                      help='Number of random combinations to run when using subset mode')
    parser.add_argument('--subset-seed', type=int, default=42,
                      help='Random seed for subset selection')
    
    # Simulation settings
    parser.add_argument('--months', type=int, default=24,
                      help='Number of months to simulate')
    parser.add_argument('--trades', type=int, default=20,
                      help='Trades per auction')
    parser.add_argument('--vessels', type=int, default=2,
                      help='Number of vessels per type')
    
    # KBestBid hyperparameters - using exact parameter names from the class
    parser.add_argument('--profit_factor', type=float, default=1.65,
                      help='Primary profit factor')
    parser.add_argument('--profit_factor_2', type=float, default=1.2,
                      help='Secondary profit factor (when bid > absolute cost)')
    parser.add_argument('--avg_w', type=float, default=0.7,
                      help='Weight for average cost in bid calculation')
    parser.add_argument('--cal_efficiency', action='store_true',
                      help='Whether to calculate efficiency for schedule sorting')
    parser.add_argument('--schedule_with_greedy', action='store_true',
                      help='Use greedy algorithm for scheduling')
    parser.add_argument('--efficiency_selection_percentage', type=float, default=0.8,
                      help='Percentage of schedules to select by efficiency')
    parser.add_argument('--trade_frequency_threshold', type=float, default=0.5,
                      help='Threshold for trade frequency')
    parser.add_argument('--k_best', type=int, default=110,
                      help='Number of schedules to generate')
    
    return parser.parse_args()

def run_simulation(args, params=None):
    """Run a single simulation with given parameters"""
    # Use provided params or fall back to args
    if params is None:
        params = {
            'profit_factor': args.profit_factor,
            'profit_factor_2': args.profit_factor_2,
            'avg_w': args.avg_w,
            'cal_efficiency': args.cal_efficiency,
            'schedule_with_greedy': args.schedule_with_greedy,
            'efficiency_selection_percentage': args.efficiency_selection_percentage,
            'trade_frequency_threshold': args.trade_frequency_threshold,
            'k_best': args.k_best
        }
    
    # Create parameter string for company name
    param_str = f"pf{params['profit_factor']}_pf2{params['profit_factor_2']}_" \
                f"avgw{params['avg_w']}_" \
                f"eff{params['cal_efficiency']}_greedy{params['schedule_with_greedy']}_" \
                f"effsel{params['efficiency_selection_percentage']}_" \
                f"freqth{params['trade_frequency_threshold']}_k{params['k_best']}"
    
    specifications_builder = environment.get_specification_builder(
        trades_per_occurrence=args.trades,
        num_auctions=args.months)
    
    # Add KBestBid company with specified parameters
    kbest_fleet = fleets.mixed_fleet(
        num_suezmax=args.vessels, 
        num_aframax=args.vessels, 
        num_vlcc=args.vessels
    )
    specifications_builder.add_company(
        kbest_bid.KBestBidComanyn.Data(
            kbest_bid.KBestBidComanyn, 
            kbest_fleet, 
            f"KBestBid_{param_str}",
            **params  # Pass all parameters 
        )
    )
    
    # Add baseline companies for comparison
    greedy_fleet = fleets.mixed_fleet(
        num_suezmax=args.vessels, 
        num_aframax=args.vessels, 
        num_vlcc=args.vessels
    )
    specifications_builder.add_company(
        greedy.GreedyComanyn.Data(
            greedy.GreedyComanyn, 
            greedy_fleet, 
            "Greedy_Baseline",
            profit_factor=1.4
        )
    )
    
    # Add standard competitors
    arch_enemy_fleet = fleets.mixed_fleet(
        num_suezmax=args.vessels, 
        num_aframax=args.vessels, 
        num_vlcc=args.vessels
    )
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, 
            arch_enemy_fleet, 
            "Arch Enemy Ltd.",
            profit_factor=1.5
        )
    )
    
    scheduler_fleet = fleets.mixed_fleet(
        num_suezmax=args.vessels, 
        num_aframax=args.vessels, 
        num_vlcc=args.vessels
    )
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, 
            scheduler_fleet, 
            "The Scheduler LP",
            profit_factor=1.4
        )
    )
    
    # Run simulation
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=False,
        global_agent_timeout=60
    )
    
    sim.run()
    



def load_sweep_config(config_path):
    """Load parameter ranges from JSON config file"""
    if not config_path:
        # Default config if none provided
        return {
            'profit_factor': [1.4, 1.5, 1.65],
            'profit_factor_2': [1.1, 1.2, 1.3],
            'avg_w': [0.5, 0.7, 0.8],
            'cal_efficiency': [False, True],
            'schedule_with_greedy': [False, True],
            'efficiency_selection_percentage': [0.8],
            'trade_frequency_threshold': [0.5, 0.6],
            'k_best': [80, 110, 150]
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def run_parameter_sweep(args):
    """Run experiments with all possible parameter combinations"""
    config = load_sweep_config(args.config)
    
    # Generate all parameter combinations using itertools.product
    import itertools
    
    # Define all parameters to sweep
    param_keys = [
        'profit_factor', 
        'profit_factor_2', 
        'avg_w', 
        'cal_efficiency', 
        'schedule_with_greedy', 
        'efficiency_selection_percentage', 
        'trade_frequency_threshold', 
        'k_best'
    ]
    
    # Get all parameter values from config
    param_values = []
    for key in param_keys:
        param_values.append(config.get(key, [getattr(args, key)]))
    
    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    
    print(f"Starting parameter sweep with {total_combinations} combinations")
    print(f"Expected runtime: {total_combinations * 5} minutes (approx. 5 min per run)")
    
    # Create results directory if it doesn't exist
    import os
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run simulations
    start_time = time.time()
    
    for i, combo in enumerate(all_combinations):
        # Create parameter dictionary for this run
        params = {param_keys[j]: combo[j] for j in range(len(param_keys))}
        
        # Create a unique identifier for this run
        run_id = f"run_{i+1}_of_{total_combinations}"
        
        print(f"\n{'-'*80}")
        print(f"Running combination {i+1}/{total_combinations}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Save hyperparameter settings to file
        params_filename = f"{results_dir}/params_{run_id}.json"
        with open(params_filename, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved to {params_filename}")
        
        # Record start time for this run
        run_start_time = time.time()
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run simulation with these parameters
        try:
            run_simulation(args, params)

        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            with open(f"{results_dir}/error_{run_id}.txt", "w") as f:
                f.write(f"Error: {str(e)}\n")
                import traceback
                traceback.print_exc(file=f)
            
        # Record end time and duration for this run
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {run_duration:.2f} seconds ({run_duration/60:.2f} minutes)")
        
        # Show progress
        elapsed_total = run_end_time - start_time
        remaining = (elapsed_total / (i+1)) * (total_combinations - i - 1)
        print(f"Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        print(f"Elapsed time: {elapsed_total/60:.1f} minutes")
        print(f"Estimated time remaining: {remaining/60:.1f} minutes")
        print(f"Estimated completion: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_end_time + remaining))}")

    
    elapsed_time = time.time() - start_time
    print(f"\nParameter sweep completed in {elapsed_time/60:.1f} minutes")
    return 

def run_parameter_subset(args):
    """Run a random subset of parameter combinations"""
    import random
    import itertools
    
    # Set random seed for reproducibility
    random.seed(args.subset_seed)
    
    # Load the full parameter space
    config = load_sweep_config(args.config)
    
    # Define all parameters
    param_keys = [
        'profit_factor', 
        'profit_factor_2', 
        'avg_w', 
        'cal_efficiency', 
        'schedule_with_greedy', 
        'efficiency_selection_percentage', 
        'trade_frequency_threshold', 
        'k_best'
    ]
    
    # Get parameter values
    param_values = []
    for key in param_keys:
        param_values.append(config.get(key, [getattr(args, key)]))
    
    # Calculate the total number of combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Total possible combinations: {total_combinations}")
    print(f"Selecting {args.subset_size} random combinations")
    
    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))
    
    # Select a random subset
    if args.subset_size < total_combinations:
        selected_combinations = random.sample(all_combinations, args.subset_size)
    else:
        print(f"Subset size {args.subset_size} exceeds total combinations {total_combinations}")
        print("Using all combinations instead")
        selected_combinations = all_combinations
    
    # Create directory for parameter configs
    params_dir = "parameter_configs"
    os.makedirs(params_dir, exist_ok=True)
    
    # Run the selected combinations
    start_time = time.time()
    
    for i, combo in enumerate(selected_combinations):
        # Create parameter dictionary
        params = {param_keys[j]: combo[j] for j in range(len(param_keys))}
        
        # Create run ID
        run_id = f"subset_{i+1}_of_{args.subset_size}"
        
        print(f"\n{'-'*80}")
        print(f"Running combination {i+1}/{args.subset_size}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Save settings
        with open(f"{params_dir}/params_{run_id}.json", "w") as f:
            json.dump(params, f, indent=2)
        
        # Run and time it
        run_start = time.time()
        try:
            # run_simulation handles results
            run_simulation(args, params)
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            with open(f"{params_dir}/error_{run_id}.txt", "w") as f:
                f.write(f"Error: {str(e)}\n")
                import traceback
                traceback.print_exc(file=f)
        
        run_duration = time.time() - run_start
        print(f"Completed in {run_duration:.2f} seconds")
    
    print(f"\nSubset experiment completed in {(time.time() - start_time)/60:.1f} minutes")

def main():
    args = parse_args()
    
    print("KBestBid Experiment Runner")
    print("-" * 40)
    
    if args.mode == 'single':
        print(f"Running single experiment with parameters:")
        print(f"  profit_factor: {args.profit_factor}")
        print(f"  profit_factor_2: {args.profit_factor_2}")
        print(f"  avg_w/abs_w: {args.avg_w}/{1-args.avg_w}")
        print(f"  cal_efficiency: {args.cal_efficiency}")
        print(f"  schedule_with_greedy: {args.schedule_with_greedy}")
        print(f"  efficiency_selection_percentage: {args.efficiency_selection_percentage}")
        print(f"  trade_frequency_threshold: {args.trade_frequency_threshold}")
        print(f"  k_best: {args.k_best}")
        
        run_simulation(args)
        
    elif args.mode == 'sweep':
        print(f"Running parameter sweep with config: {args.config or 'default'}")
        run_parameter_sweep(args)
        
    elif args.mode == 'subset':
        print(f"Running parameter subset with config: {args.config or 'default'}")
        run_parameter_subset(args)
        
    print("\nExperiment completed!")

if __name__ == '__main__':
    main() 