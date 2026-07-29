import json
import torch
import os
import gc
import time
import math
import argparse
import random
from pathlib import Path

import numpy as np

from src.trainer import train_model, get_final_raw_solution
from src.data_reading import (
    read_set_cover_instance,
    read_subset_sum_instance,
    read_hypermaxcut_instance,
    read_hitting_set_instance,
    generate_incidence_matrix
)
from src.model import ImprovedHyperGraphNet
from src.utils import (
    check_problem_feasibility,
    validate_solution,
    integrated_post_process_set_cover,
    integrated_post_process_hitting_set,
    post_process_solution_hypermultiwaycut,
    post_process_solution_hypermaxcut,
    post_process_solution_subset_sum
)
from src.wandb_logger import RunLogger


BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "./data")).resolve()
if "BENCHMARK_ROOT" not in os.environ:
    print(f"[Warning] BENCHMARK_ROOT not set; defaulting to {BENCHMARK_ROOT}. "
          f"Set BENCHMARK_ROOT for cross-method instance_id matching in future evals.")


def compute_instance_id(file_path):
    """
    instance_id per wandb_tracking_spec.md Section 5: the loaded file's absolute
    path, made relative to BENCHMARK_ROOT, with forward slashes.
    """
    resolved = Path(file_path).resolve()
    try:
        return str(resolved.relative_to(BENCHMARK_ROOT)).replace("\\", "/")
    except ValueError:
        print(f"[Warning] {resolved} is not under BENCHMARK_ROOT={BENCHMARK_ROOT}; "
              f"using bare filename as instance_id.")
        return resolved.name


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_nn_analysis(stats, problem_type="set_cover"):
    """
    Print info from the dictionary returned by get_final_raw_solution(...).
    """
    print("\n-- Raw Neural Network Solution Analysis --")

    if problem_type == "set_cover":
        cov = stats['coverage_ratio']
        print(f"NN solution size:        {int(stats['size'])}")
        print(f"NN solution validity:    {stats['valid']}")
        print(f"NN coverage ratio:       {cov:.4f}")
        print(f"NN mean prob:            {stats['mean_prob']:.4f}")
        print(f"NN std prob:             {stats['std_prob']:.4f}")
        print(f"Dynamic threshold numeric value: {stats.get('dynamic_thr_value','(N/A)')}")
        print(f"Threshold=0.5 numeric value:    {stats.get('threshold_05_value','(N/A)')}")

    elif problem_type == "hitting_set":
        cov = stats['coverage_ratio']
        print(f"NN solution size (elements selected): {int(stats['size'])}")
        print(f"NN solution validity:    {stats['valid']}")
        print(f"NN coverage ratio:       {cov:.4f}")
        print(f"NN mean prob:            {stats['mean_prob']:.4f}")
        print(f"NN std prob:             {stats['std_prob']:.4f}")

    elif problem_type == "subset_sum":
        ratio = stats['coverage_ratio']
        print(f"NN solution size:        {int(stats['size'])}")
        print(f"NN ratio to target sum:  {ratio:.4f}")
        print(f"NN mean prob:            {stats['mean_prob']:.4f}")
        print(f"NN std prob:             {stats['std_prob']:.4f}")
        if 'threshold' in stats:
            print(f"Threshold used: {stats['threshold']:.4f}")

    elif problem_type == "hypermaxcut":
        dyn_raw_cov = stats.get('coverage_ratio_dyn_raw', 0.0)
        thr05_raw_cov = stats.get('coverage_ratio_05_raw', 0.0)
        print(f"Fraction of hyperedges cut (dynamic threshold, raw):  {dyn_raw_cov:.4f}")
        print(f"Fraction of hyperedges cut (0.5 threshold, raw):       {thr05_raw_cov:.4f}")
        print(f"\nChosen best post-processed => fraction cut:  {stats['coverage_ratio']:.4f}")
        print(f"Size of side=1 (best solution): {int(stats['size'])}")
        print(f"Mean prob: {stats['mean_prob']:.4f}, Std prob: {stats['std_prob']:.4f}")
        print(f"Which threshold used: {stats.get('which_threshold_was_used', 'N/A')}")
        print(f"Raw coverage ratio before refinement: {stats.get('raw_chosen_coverage_ratio',0.0):.4f}")
        if 'dynamic_thr_value' in stats:
            print(f"Dynamic threshold numeric value: {stats['dynamic_thr_value']:.4f}")
        if 'threshold_05_value' in stats:
            print(f"Threshold=0.5 numeric value: {stats['threshold_05_value']:.4f}")

    elif problem_type == "hypermultiwaycut":
        dyn_raw_cov = stats.get('coverage_ratio_dyn_raw', 0.0)
        thr05_raw_cov = stats.get('coverage_ratio_05_raw', 0.0)
        print(f"Fraction of hyperedges cut (argmax dynamic, raw):  {dyn_raw_cov:.4f}")
        print(f"Fraction of hyperedges cut (0.5 threshold, raw):    {thr05_raw_cov:.4f}")
        print(f"\nFinal post-processed coverage ratio: {stats['coverage_ratio']:.4f}")
        print(f"Which approach used: {stats.get('which_raw_solution_used', 'N/A')}")
        print(f"Raw coverage ratio chosen (before refinement): {stats.get('raw_chosen_coverage_ratio',0.0):.4f}")
        print(f"Mean prob across partitions: {stats['mean_prob']:.4f}")
        print(f"Std prob: {stats['std_prob']:.4f}")


def parse_shape_set_cover(file_path):
    subsets, elements, hdr = read_set_cover_instance(file_path)
    n_elems = int(hdr[0])
    n_subs = int(hdr[1])
    return (n_elems, n_subs)


def parse_shape_hitting_set(file_path):
    subsets, elements, hdr = read_hitting_set_instance(file_path)
    num_sets = int(hdr[0])
    num_elems = int(hdr[1])
    return (num_elems, num_sets)


def parse_shape_subset_sum(file_path):
    subs, elems, hdr, weights = read_subset_sum_instance(file_path)
    n_elems = int(hdr[0])
    n_subs = int(hdr[1])
    return (n_elems, n_subs)


def parse_shape_hypermaxcut(file_path):
    subs, elems, hdr = read_hypermaxcut_instance(file_path)
    n_nodes = int(hdr[0])
    n_edges = int(hdr[1])
    return (n_nodes, n_edges)


def parse_shape_hypermultiwaycut(file_path, config):
    subs, elems, hdr = read_hypermaxcut_instance(file_path)
    n_nodes = int(hdr[0])
    n_edges = int(hdr[1])
    k = config.get('num_partitions', 4)
    return (n_nodes, n_edges, k)


def build_model_for_shape(problem_type, shape, params):
    """
    Construct an ImprovedHyperGraphNet for the given shape.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 128
    num_layers = 4
    dropout_rate = 0.1
    num_partitions = params.get('num_partitions', 2)

    if problem_type in ["set_cover", "subset_sum", "hypermaxcut"]:
        if len(shape) == 2:
            n_elem, n_sub = shape
            return ImprovedHyperGraphNet(
                num_elements=n_elem,
                num_subsets=n_sub,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                num_partitions=num_partitions,
                problem_type=problem_type
            ).to(dev)

    elif problem_type == "hitting_set":
        if len(shape) == 2:
            n_elem, n_sets = shape
            return ImprovedHyperGraphNet(
                num_elements=n_elem,
                num_subsets=n_sets,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                num_partitions=num_partitions,
                problem_type="hitting_set"
            ).to(dev)

    elif problem_type == "hypermultiwaycut":
        if len(shape) == 3:
            n_nodes, n_edges, k = shape
            return ImprovedHyperGraphNet(
                num_elements=n_nodes,
                num_subsets=n_edges,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                num_partitions=k,
                problem_type="hypermultiwaycut"
            ).to(dev)

    return None


def process_set_cover_instance(file_path, params, model=None, skip_train=False, reference_shape=None):
    try:
        subsets, elements, header = read_set_cover_instance(file_path)
        num_elements = int(header[0])
        num_subsets = int(header[1])
        print(f"Instance size: {num_elements} elements, {num_subsets} subsets\n")

        if reference_shape is not None and reference_shape != (num_elements, num_subsets):
            print("  [Shape mismatch: skipping file.]")
            return None

        inc_mat = generate_incidence_matrix(subsets, elements).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inc_mat = inc_mat.to(device)

        if not check_problem_feasibility(inc_mat, "set_cover"):
            print("Instance is infeasible.")
            return None

        if model is None:
            model = build_model_for_shape("set_cover", (num_elements, num_subsets), params)

        train_start = time.time()
        if not skip_train:
            best_probs = train_model(model, inc_mat, params, problem_type="set_cover", extra_info=None)
        else:
            model.eval()
            with torch.no_grad():
                out_ = model(inc_mat)
                out_ = out_.clamp(1e-3, 1 - 1e-3)
            best_probs = out_
        training_time = time.time() - train_start

        map_start = time.time()
        nn_stats = get_final_raw_solution(best_probs, inc_mat, problem_type="set_cover", extra_info=None)
        total_get_sol_time = time.time() - map_start

        raw_post_time = nn_stats.get('raw_post_time', 0.0)
        mapping_time = max(0.0, total_get_sol_time - raw_post_time)

        print_nn_analysis(nn_stats, "set_cover")
        raw_selected = (nn_stats['solution'] >= 0.5).nonzero(as_tuple=True)[0].tolist()
        print(f"NN subsets selected (raw): {raw_selected}")
        print(f"NN raw solution size:      {nn_stats['solution'].sum().item()}")

        post_time = 0.0
        final_sol = nn_stats['solution'].clone()
        final_val = validate_solution(final_sol, inc_mat, "set_cover")

        if not skip_train:
            post_start = time.time()
            final_sol = integrated_post_process_set_cover(best_probs, inc_mat)
            external_time = time.time() - post_start
            post_time = raw_post_time + external_time

            final_sz = int((final_sol >= 0.5).sum().item())
            final_val = validate_solution(final_sol, inc_mat, "set_cover")
            print("\n-- Final Post-Processed Solution --")
            print(f"Final solution size:      {final_sz}")
            print(f"Final solution validity:  {'Valid' if final_val else 'Invalid'}")
            sel_idx = (final_sol >= 0.5).nonzero(as_tuple=True)[0].tolist()
            print(f"Final subsets selected:   {sel_idx}\n")
        else:
            print("\n[Skip post-processing because skip_train=True]\n")

        return {
            'solution': final_sol,
            'valid': final_val,
            'model': model,
            'shape': (num_elements, num_subsets),
            'training_time': training_time,
            'mapping_time': mapping_time,
            'post_time': post_time,
            'total_time': training_time + mapping_time + post_time
        }

    except Exception as e:
        print(f"Error processing instance: {e}")
        return None


def process_hitting_set_instance(file_path, params, model=None, skip_train=False, reference_shape=None):
    try:
        subsets, elements, header = read_hitting_set_instance(file_path)
        num_sets = int(header[0])
        num_elements = int(header[1])
        print(f"Instance size: {num_sets} sets, {num_elements} elements\n")

        if reference_shape is not None and reference_shape != (num_elements, num_sets):
            print("  [Shape mismatch: skipping file.]")
            return None

        inc_mat = generate_incidence_matrix(subsets, elements).float().t()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inc_mat = inc_mat.to(dev)

        if not check_problem_feasibility(inc_mat, "hitting_set"):
            print("Hitting set instance infeasible.")
            return None

        if model is None:
            model = build_model_for_shape("hitting_set", (num_elements, num_sets), params)

        train_start = time.time()
        if not skip_train:
            best_probs = train_model(model, inc_mat, params, problem_type="hitting_set", extra_info=None)
        else:
            model.eval()
            with torch.no_grad():
                out_ = model(inc_mat)
                out_ = out_.clamp(1e-3, 1 - 1e-3)
            best_probs = out_
        training_time = time.time() - train_start

        map_start = time.time()
        nn_stats = get_final_raw_solution(best_probs, inc_mat, problem_type="hitting_set", extra_info=None)
        total_get_sol_time = time.time() - map_start

        raw_post_time = nn_stats.get('raw_post_time', 0.0)
        mapping_time = max(0.0, total_get_sol_time - raw_post_time)

        print_nn_analysis(nn_stats, "hitting_set")
        raw_selected = (nn_stats['solution'] >= 0.5).nonzero(as_tuple=True)[0].tolist()
        print(f"NN elements selected (raw): {raw_selected}")
        print(f"NN raw solution size:      {nn_stats['solution'].sum().item()}")

        post_time = 0.0
        final_sol = nn_stats['solution'].clone()
        final_val = validate_solution(final_sol, inc_mat, "hitting_set")

        if not skip_train:
            post_start = time.time()
            final_sol = integrated_post_process_hitting_set(best_probs, inc_mat)
            external_time = time.time() - post_start
            post_time = raw_post_time + external_time

            final_sz = int((final_sol >= 0.5).sum().item())
            final_val = validate_solution(final_sol, inc_mat, "hitting_set")
            print("\n-- Final Post-Processed Hitting Set Solution --")
            print(f"Final solution size (elements selected): {final_sz}")
            print(f"Final solution validity:  {'Valid' if final_val else 'Invalid'}")
            sel_idx = (final_sol >= 0.5).nonzero(as_tuple=True)[0].tolist()
            print(f"Final elements selected:   {sel_idx}\n")
        else:
            print("\n[Skip post-processing because skip_train=True]\n")

        return {
            'solution': final_sol,
            'valid': final_val,
            'model': model,
            'shape': (num_elements, num_sets),
            'training_time': training_time,
            'mapping_time': mapping_time,
            'post_time': post_time,
            'total_time': training_time + mapping_time + post_time
        }

    except Exception as e:
        print(f"Error processing hitting set instance: {e}")
        return None


def process_subset_sum_instance(file_path, params, model=None, skip_train=False, reference_shape=None):
    try:
        subsets, elements, header, weights = read_subset_sum_instance(file_path)
        num_elements = int(header[0])
        num_subsets = int(header[1])
        target_sum = float(header[2])
        print(f"Subset Sum: {num_elements} items, {num_subsets} hyperedge(s), target_sum={target_sum}\n")

        if reference_shape is not None and reference_shape != (num_elements, num_subsets):
            print("  [Shape mismatch: skipping file.]")
            return None

        inc_mat = generate_incidence_matrix(subsets, elements).float()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inc_mat = inc_mat.to(dev)

        feasible = check_problem_feasibility(inc_mat, "subset_sum", (weights, target_sum))
        if not feasible:
            print("Subset Sum infeasible => sum(weights) < target?")
            return None

        if model is None:
            model = build_model_for_shape("subset_sum", (num_elements, num_subsets), params)

        train_start = time.time()
        if not skip_train:
            best_probs = train_model(model, inc_mat, params, problem_type="subset_sum", extra_info=(weights, target_sum))
        else:
            model.eval()
            with torch.no_grad():
                out_ = model(inc_mat)
                out_ = out_.clamp(1e-3, 1 - 1e-3)
            best_probs = out_
        training_time = time.time() - train_start

        map_start = time.time()
        nn_stats = get_final_raw_solution(best_probs, inc_mat, "subset_sum", extra_info=(weights, target_sum))
        total_get_sol_time = time.time() - map_start

        raw_post_time = nn_stats.get('raw_post_time', 0.0)
        mapping_time = max(0.0, total_get_sol_time - raw_post_time)

        print_nn_analysis(nn_stats, "subset_sum")

        post_time = 0.0
        final_sol = nn_stats['solution'].clone()
        final_val = validate_solution(final_sol, inc_mat, "subset_sum", (weights, target_sum))

        if not skip_train:
            post_start = time.time()
            final_sol, final_diff = post_process_solution_subset_sum(best_probs, weights, target_sum)
            external_time = time.time() - post_start
            post_time = raw_post_time + external_time

            final_val = validate_solution(final_sol, inc_mat, "subset_sum", (weights, target_sum))
            sel_idx = (final_sol >= 0.5).nonzero(as_tuple=True)[0].tolist()
            
            w_t = torch.tensor(weights, device=final_sol.device)
            final_sum = float((final_sol * w_t).sum().item())
            final_deviation = abs(final_sum - target_sum)
            
            print("\n-- Post-Processed Subset Sum Solution --")
            print(f"Items chosen = {sel_idx}")
            print(f"Selected items: {int(final_sol.sum().item())}/{num_elements}")
            print(f"Final sum achieved = {final_sum:.1f}, target = {target_sum:.1f}")
            print(f"Final deviation = {final_deviation:.1f}, valid = {final_val}\n")
        else:
            print("\n[Skip post-processing because skip_train=True]\n")

        return {
            'solution': final_sol,
            'valid': final_val,
            'model': model,
            'shape': (num_elements, num_subsets),
            'training_time': training_time,
            'mapping_time': mapping_time,
            'post_time': post_time,
            'total_time': training_time + mapping_time + post_time
        }

    except Exception as e:
        print(f"Error processing subset sum instance: {e}")
        return None


def process_hypermaxcut_instance(file_path, params, model=None, skip_train=False, reference_shape=None, seed=0):
    # Constructed before the try/except below so a missing WANDB_API_KEY fails
    # loudly instead of being swallowed as "Error processing hypermaxcut instance".
    logger = RunLogger()
    set_all_seeds(seed)

    try:
        subsets, elements, header = read_hypermaxcut_instance(file_path)
        num_nodes = int(header[0])
        num_hypered = int(header[1])
        print(f"HyperMaxCut: {num_nodes} nodes, {num_hypered} hyperedges\n")

        if reference_shape is not None and reference_shape != (num_nodes, num_hypered):
            print("  [Shape mismatch: skipping file.]")
            return None

        difficulty_param = (sum(len(s) for s in subsets) / len(subsets)) / num_nodes if subsets else 0.0
        logger.start({
            "method": "ours",
            "problem_type": "maxcut",
            "instance_id": compute_instance_id(file_path),
            "instance_size": num_nodes,
            "difficulty_param": difficulty_param,
            "seed": seed,
            "condition": "n/a",
            "pretrain_scale": "n/a",
            "time_budget_s": None,
        })

        try:
            inc_mat = generate_incidence_matrix(subsets, elements).float()
            inc_mat = inc_mat.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            feasible = check_problem_feasibility(inc_mat, "hypermaxcut")
            if not feasible:
                print("HyperMaxCut instance infeasible.")
                return None

            if model is None:
                model = build_model_for_shape("hypermaxcut", (num_nodes, num_hypered), params)

            train_start = time.time()
            if not skip_train:
                best_probs = train_model(model, inc_mat, params, problem_type="hypermaxcut",
                                         extra_info=None, logger=logger)
            else:
                model.eval()
                with torch.no_grad():
                    out_ = model(inc_mat)
                    out_ = out_.clamp(1e-3, 1 - 1e-3)
                best_probs = out_
            training_time = time.time() - train_start

            map_start = time.time()
            nn_stats = get_final_raw_solution(best_probs, inc_mat, "hypermaxcut", extra_info=None)
            total_get_sol_time = time.time() - map_start

            raw_post_time = nn_stats.get('raw_post_time', 0.0)
            mapping_time = max(0.0, total_get_sol_time - raw_post_time)

            print_nn_analysis(nn_stats, "hypermaxcut")

            post_time = 0.0
            final_sol = nn_stats['solution'].clone()
            final_val = True
            post_ratio = nn_stats['coverage_ratio']

            if not skip_train:
                post_start = time.time()
                extra_pp_time = time.time() - post_start
                post_time = raw_post_time + extra_pp_time

                print("\n-- Post-Processed HyperMaxCut Solution --")
                print(f"Fraction of hyperedges cut (post-processed): {post_ratio:.4f}")
                print(f"Size (side=1): {(final_sol >= 0.5).sum().item()}")
                sel_idx = (final_sol >= 0.5).nonzero(as_tuple=True)[0].tolist()
                print(f"Nodes in side=1: {sel_idx}\n")
            else:
                print("\n[Skip post-processing because skip_train=True]\n")

            logger.log_step("post_refinement", nn_stats['coverage_ratio'])
            logger.set_summary(
                t_pre_refinement_total_s=training_time,
                t_post_refinement_total_s=raw_post_time,
                final_quality_raw=nn_stats.get('raw_chosen_coverage_ratio', nn_stats['coverage_ratio']),
                final_quality_refined=nn_stats['coverage_ratio'],
            )

            return {
                'solution': final_sol,
                'valid': final_val,
                'model': model,
                'shape': (num_nodes, num_hypered),
                'training_time': training_time,
                'mapping_time': mapping_time,
                'post_time': post_time,
                'total_time': training_time + mapping_time + post_time
            }
        finally:
            logger.finish()

    except Exception as e:
        print(f"Error processing hypermaxcut instance: {e}")
        return None


def process_hypermultiwaycut_instance(file_path, params, model=None, skip_train=False, reference_shape=None):
    try:
        subsets, elements, header = read_hypermaxcut_instance(file_path)
        num_nodes = int(header[0])
        num_hypered = int(header[1])
        k = params.get('num_partitions', 4)
        print(f"HyperMultiwayCut: {num_nodes} nodes, {num_hypered} hyperedges, partitions={k}\n")

        if reference_shape is not None and reference_shape != (num_nodes, num_hypered):
            print("  [Shape mismatch: skipping file.]")
            return None

        inc_mat = generate_incidence_matrix(subsets, elements).float()
        inc_mat = inc_mat.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        feasible = check_problem_feasibility(inc_mat, "hypermultiwaycut")
        if not feasible:
            print("HyperMultiwayCut instance infeasible.")
            return None

        if model is None:
            model = build_model_for_shape("hypermultiwaycut", (num_nodes, num_hypered, k), params)

        train_start = time.time()
        if not skip_train:
            best_probs = train_model(model, inc_mat, params, problem_type="hypermultiwaycut", extra_info=None)
        else:
            model.eval()
            with torch.no_grad():
                best_probs = model(inc_mat)
        training_time = time.time() - train_start

        map_start = time.time()
        nn_stats = get_final_raw_solution(best_probs, inc_mat, "hypermultiwaycut", extra_info=None)
        total_get_sol_time = time.time() - map_start

        raw_post_time = nn_stats.get('raw_post_time', 0.0)
        mapping_time = max(0.0, total_get_sol_time - raw_post_time)

        print_nn_analysis(nn_stats, "hypermultiwaycut")

        post_time = 0.0
        final_sol = nn_stats['solution'].clone()
        final_val = True
        if not skip_train:
            post_start = time.time()
            external_pp_time = time.time() - post_start
            post_time = raw_post_time + external_pp_time

            print("\n-- Post-Processed HyperMultiwayCut Solution --")
        else:
            print("\n[Skip post-processing because skip_train=True]\n")

        return {
            'solution': final_sol,
            'valid': final_val,
            'model': model,
            'shape': (num_nodes, num_hypered),
            'training_time': training_time,
            'mapping_time': mapping_time,
            'post_time': post_time,
            'total_time': training_time + mapping_time + post_time
        }

    except Exception as e:
        print(f"Error processing hypermultiwaycut instance file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str,
                        choices=["set_cover", "subset_sum", "hypermaxcut",
                                 "hypermultiwaycut", "hitting_set"],
                        default="set_cover",
                        help="Which problem type to run.")
    parser.add_argument("--mode", type=str,
                        choices=["instance_specific","pretrain","test_only","test_finetune"],
                        default="instance_specific",
                        help="Mode to run.")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path to save/load model state dict.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (hypermaxcut only; used for the W&B "
                             "10-independent-runs-per-instance methodology).")
    args = parser.parse_args()

    if args.problem == "subset_sum":
        config_path = 'configs/subset_sum_config.json'
    elif args.problem == "hypermaxcut":
        config_path = 'configs/hypermaxcut_config.json'
    elif args.problem == "hypermultiwaycut":
        config_path = 'configs/hypermultiwaycut_config.json'
    elif args.problem == "hitting_set":
        config_path = 'configs/hitting_set_config.json'
    else:
        config_path = 'configs/set_cover_config.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        print(f"[Warning] Could not load config file {config_path}, using defaults.")
        config = None

    def initialize_parameters(conf):
        default_params = {
            'lr': 0.0005,
            'min_epochs': 30,
            'max_epochs': 200,
            'patience': 10,
            'weight_decay': 0.0005,
            'coverage_floor': 0.4,
            'coverage_floor_patience': 15,
            'phase2_cov': 0.98,
            'phase2_patience': 3,
            'tol': 1e-4
        }
        if not conf:
            return default_params
        base_config = conf.get('training', {}).get('base', {})
        params = {**default_params, **base_config}
        params['phase2_cov'] = conf.get('training', {}).get('phase2_cov', default_params['phase2_cov'])
        params['phase2_patience'] = conf.get('training', {}).get('phase2_patience', default_params['phase2_patience'])
        return params

    params = initialize_parameters(config)
    if config and 'training' in config and 'num_partitions' in config['training']:
        params['num_partitions'] = config['training']['num_partitions']

    if args.problem == "subset_sum":
        folder_path = config.get('folder_path', './data/subset_sum/') if config else './data/subset_sum/'
    elif args.problem == "hypermaxcut":
        folder_path = config.get('folder_path', './data/hypergraph_data/') if config else './data/hypergraph_data/'
    elif args.problem == "hypermultiwaycut":
        folder_path = config.get('folder_path', './data/hypergraph_data_multi/') if config else './data/hypergraph_data_multi/'
    elif args.problem == "hitting_set":
        folder_path = config.get('folder_path', './data/set_cover/') if config else './data/set_cover/'
    else:
        folder_path = config.get('folder_path', './data/set_cover/') if config else './data/set_cover/'

    files = sorted(fn for fn in os.listdir(folder_path) if fn.endswith('.txt') and not fn.startswith('.'))

    results = {}
    total_training_time_all = 0.0
    total_mapping_time_all = 0.0
    total_post_time_all = 0.0
    total_start_time = time.time()

    shape2model_state = {}

    def make_shape_key(shape, model):
        """
        If shape is 2D => (n, s, hidden_dim).
        If shape is 3D => (n, e, k, hidden_dim).
        """
        hd = model.hypergraph_net.hidden_dim
        if len(shape) == 3:
            n, e, k = shape
            return (n, e, k, hd)
        elif len(shape) == 2:
            n, s = shape
            return (n, s, hd)
        else:
            return None

    def parse_shape_key(k):
        """
        If len(k) == 4 => (n,e,k,hd).
        Else => (n,s,hd).
        Return (shape, hidden_dim, is3d).
        """
        if len(k) == 4:
            return (k[0], k[1], k[2]), k[3], True
        else:
            return (k[0], k[1]), k[2], False

    def find_closest_shape(target_shape, shape_states):
        """
        Only compare dimension = 2D or 3D.
        If 3D => must match k exactly, measure distance in (n+e).
        If 2D => measure distance in (n+s).
        """
        best_key = None
        best_dist = 1e15
        is_t3d = (len(target_shape) == 3)
        if is_t3d:
            sum_target = target_shape[0] + target_shape[1]
            want_k = target_shape[2]
        else:
            sum_target = target_shape[0] + target_shape[1]

        for k_ in shape_states.keys():
            shp_, hidden_dim_, is_3d_ = parse_shape_key(k_)
            if is_3d_ != is_t3d:
                continue
            if is_3d_:
                if shp_[2] != want_k:
                    continue
                sum_k_ = shp_[0] + shp_[1]
                dist_ = abs(sum_k_ - sum_target)
            else:
                sum_k_ = shp_[0] + shp_[1]
                dist_ = abs(sum_k_ - sum_target)

            if dist_ < best_dist:
                best_dist = dist_
                best_key = k_
        return best_key

    if args.mode == "pretrain":
        if len(files) == 0:
            print(f"No .txt files found in folder: {folder_path}")
            return

        shape2model_trained = {}

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            print(f"\n[Pretrain Mode] Training on file: {fname}")
            if args.problem == "set_cover":
                shp = parse_shape_set_cover(fpath)
                out = process_set_cover_instance(fpath, params, None, skip_train=False)
            elif args.problem == "hitting_set":
                shp = parse_shape_hitting_set(fpath)
                out = process_hitting_set_instance(fpath, params, None, skip_train=False)
            elif args.problem == "subset_sum":
                shp = parse_shape_subset_sum(fpath)
                out = process_subset_sum_instance(fpath, params, None, skip_train=False)
            elif args.problem == "hypermaxcut":
                shp = parse_shape_hypermaxcut(fpath)
                out = process_hypermaxcut_instance(fpath, params, None, skip_train=False, seed=args.seed)
            else:
                shp = parse_shape_hypermultiwaycut(fpath, config)
                out = process_hypermultiwaycut_instance(fpath, params, None, skip_train=False)

            if shp is None or out is None:
                continue
            if 'model' in out and out['model'] is not None:
                shape2model_trained[shp] = out['model']
                total_training_time_all += out.get('training_time', 0)
                total_mapping_time_all += out.get('mapping_time', 0)
                total_post_time_all += out.get('post_time', 0)

        all_states = {}
        for shp_, model_ in shape2model_trained.items():
            if model_ is None:
                continue
            key_ = make_shape_key(shp_, model_)
            if key_ is None:
                continue
            st_dict = model_.state_dict()
            all_states[key_] = {'state': st_dict}

        if len(all_states) == 0:
            print("No trained models found. Possibly no valid files.")
        else:
            outpath = args.pretrained_model_path if args.pretrained_model_path else f"pretrained_{args.problem}_bank.pt"
            torch.save(all_states, outpath)
            print(f"[Pretrain] Model bank saved to {outpath}")

    elif args.mode == "test_only":
        if not os.path.isfile(args.pretrained_model_path):
            print(f"[Test-Only] No valid pretrained model file: {args.pretrained_model_path}")
            return

        print(f"[Test-Only] Loading shape-based model bank from {args.pretrained_model_path}")
        loaded_dict = torch.load(args.pretrained_model_path,
                                 map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for k_ in loaded_dict.keys():
            shape2model_state[k_] = loaded_dict[k_]

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            if args.problem == "set_cover":
                shp = parse_shape_set_cover(fpath)
                out = None
            elif args.problem == "hitting_set":
                shp = parse_shape_hitting_set(fpath)
                out = None
            elif args.problem == "subset_sum":
                shp = parse_shape_subset_sum(fpath)
                out = None
            elif args.problem == "hypermaxcut":
                shp = parse_shape_hypermaxcut(fpath)
                out = None
            else:
                shp = parse_shape_hypermultiwaycut(fpath, config)
                out = None

            if shp is None:
                print(f"\n[Test-Only] {fname}: parse_shape returned None => skipping.")
                continue

            new_model = build_model_for_shape(args.problem, shp, params)
            if new_model is None:
                print(f"[Test-Only] {fname}: shape {shp} => can't build model => skipping.")
                continue

            shape_key = make_shape_key(shp, new_model)
            if shape_key in shape2model_state:
                st_info = shape2model_state[shape_key]
                new_model.load_state_dict(st_info['state'], strict=True)
                print(f"[Test-Only] Found perfect shape match for {fname}.")
            else:
                best_key = find_closest_shape(shp, shape2model_state)
                if best_key is not None:
                    print(f"[Test-Only] Using partial shape reuse from {best_key} => {fname}.")
                    st_info = shape2model_state[best_key]
                    from copy import deepcopy
                    old_shp, old_hd, old_is3d = parse_shape_key(best_key)
                    if old_is3d:
                        base_model = ImprovedHyperGraphNet(
                            old_shp[0], old_shp[1],
                            hidden_dim=old_hd, 
                            num_layers=4,
                            dropout_rate=0.1,
                            num_partitions=old_shp[2],
                            problem_type=args.problem
                        ).to(new_model.hypergraph_net.node_embedding.device)
                    else:
                        base_model = ImprovedHyperGraphNet(
                            old_shp[0], old_shp[1],
                            hidden_dim=old_hd,
                            num_layers=4,
                            dropout_rate=0.1,
                            num_partitions=2,
                            problem_type=args.problem
                        ).to(new_model.hypergraph_net.node_embedding.device)
                    base_model.load_state_dict(st_info['state'], strict=True)
                    base_model.eval()
                    with torch.no_grad():
                        new_dict = new_model.state_dict()
                        base_dict = base_model.state_dict()
                        skip_keys = ["hypergraph_net.node_embedding",
                                     "hypergraph_net.edge_embedding"]
                        for kparam in base_dict.keys():
                            if kparam in skip_keys:
                                continue
                            if kparam in new_dict and new_dict[kparam].shape == base_dict[kparam].shape:
                                new_dict[kparam] = base_dict[kparam].clone()
                        new_model.load_state_dict(new_dict, strict=False)
                else:
                    print(f"[Test-Only] No suitable pretrained shape => random fallback for {fname}.")

            if args.problem == "set_cover":
                out = process_set_cover_instance(fpath, params, new_model, skip_train=True)
            elif args.problem == "hitting_set":
                out = process_hitting_set_instance(fpath, params, new_model, skip_train=True)
            elif args.problem == "subset_sum":
                out = process_subset_sum_instance(fpath, params, new_model, skip_train=True)
            elif args.problem == "hypermaxcut":
                out = process_hypermaxcut_instance(fpath, params, new_model, skip_train=True, seed=args.seed)
            else:
                out = process_hypermultiwaycut_instance(fpath, params, new_model, skip_train=True)

            if out is not None:
                results[fname] = out
                total_training_time_all += out['training_time']
                total_mapping_time_all += out['mapping_time']
                total_post_time_all += out['post_time']

    elif args.mode == "test_finetune":
        if not os.path.isfile(args.pretrained_model_path):
            print(f"[Test-FineTune] No valid pretrained model file: {args.pretrained_model_path}")
            return
        print(f"[Test-FineTune] Loading shape-based model bank from {args.pretrained_model_path}")
        loaded_dict = torch.load(args.pretrained_model_path,
                                 map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for k_ in loaded_dict.keys():
            shape2model_state[k_] = loaded_dict[k_]

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            if args.problem == "set_cover":
                shp = parse_shape_set_cover(fpath)
            elif args.problem == "hitting_set":
                shp = parse_shape_hitting_set(fpath)
            elif args.problem == "subset_sum":
                shp = parse_shape_subset_sum(fpath)
            elif args.problem == "hypermaxcut":
                shp = parse_shape_hypermaxcut(fpath)
            else:
                shp = parse_shape_hypermultiwaycut(fpath, config)

            if shp is None:
                print(f"\n[Test-FineTune] {fname}: shape parse error => skipping.")
                continue

            new_model = build_model_for_shape(args.problem, shp, params)
            if new_model is None:
                print(f"[Test-FineTune] {fname}: can't build model => skip.")
                continue

            shape_key = make_shape_key(shp, new_model)
            if shape_key in shape2model_state:
                st_info = shape2model_state[shape_key]
                new_model.load_state_dict(st_info['state'], strict=True)
                print(f"[Test-FineTune] Found perfect shape match for {fname}.")
            else:
                best_key = find_closest_shape(shp, shape2model_state)
                if best_key is not None:
                    old_shp, old_hd, old_is3d = parse_shape_key(best_key)
                    st_info = shape2model_state[best_key]
                    if old_is3d:
                        base_model = ImprovedHyperGraphNet(
                            old_shp[0], old_shp[1],
                            hidden_dim=old_hd,
                            num_layers=4,
                            dropout_rate=0.1,
                            num_partitions=old_shp[2],
                            problem_type=args.problem
                        ).to(new_model.hypergraph_net.node_embedding.device)
                    else:
                        base_model = ImprovedHyperGraphNet(
                            old_shp[0], old_shp[1],
                            hidden_dim=old_hd,
                            num_layers=4,
                            dropout_rate=0.1,
                            num_partitions=2,
                            problem_type=args.problem
                        ).to(new_model.hypergraph_net.node_embedding.device)

                    base_model.load_state_dict(st_info['state'], strict=True)
                    base_model.eval()
                    with torch.no_grad():
                        new_dict = new_model.state_dict()
                        base_dict = base_model.state_dict()
                        skip_keys = ["hypergraph_net.node_embedding",
                                     "hypergraph_net.edge_embedding"]
                        for kparam in base_dict.keys():
                            if kparam in skip_keys:
                                continue
                            if (kparam in new_dict) and (new_dict[kparam].shape == base_dict[kparam].shape):
                                new_dict[kparam] = base_dict[kparam].clone()
                        new_model.load_state_dict(new_dict, strict=False)
                    print(f"[Test-FineTune] Using partial reuse from shape={best_key} => {fname}.")
                else:
                    print(f"[Test-FineTune] no suitable shape => fallback for {fname}.")

            old_max = params['max_epochs']
            old_pat = params['patience']
            params['max_epochs'] = 60
            params['patience'] = 10

            if args.problem == "set_cover":
                out = process_set_cover_instance(fpath, params, new_model, skip_train=False)
            elif args.problem == "hitting_set":
                out = process_hitting_set_instance(fpath, params, new_model, skip_train=False)
            elif args.problem == "subset_sum":
                out = process_subset_sum_instance(fpath, params, new_model, skip_train=False)
            elif args.problem == "hypermaxcut":
                out = process_hypermaxcut_instance(fpath, params, new_model, skip_train=False, seed=args.seed)
            else:
                out = process_hypermultiwaycut_instance(fpath, params, new_model, skip_train=False)

            params['max_epochs'] = old_max
            params['patience'] = old_pat

            if out is not None:
                results[fname] = out
                total_training_time_all += out['training_time']
                total_mapping_time_all += out['mapping_time']
                total_post_time_all += out['post_time']

    else:
        for fname in files:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()
            fpath = os.path.join(folder_path, fname)

            if args.problem == "set_cover":
                out = process_set_cover_instance(fpath, params, None, skip_train=False)
            elif args.problem == "hitting_set":
                out = process_hitting_set_instance(fpath, params, None, skip_train=False)
            elif args.problem == "subset_sum":
                out = process_subset_sum_instance(fpath, params, None, skip_train=False)
            elif args.problem == "hypermaxcut":
                out = process_hypermaxcut_instance(fpath, params, None, skip_train=False, seed=args.seed)
            else:
                out = process_hypermultiwaycut_instance(fpath, params, None, skip_train=False)

            if out is not None:
                results[fname] = out
                total_training_time_all += out['training_time']
                total_mapping_time_all += out['mapping_time']
                total_post_time_all += out['post_time']

    total_time = time.time() - total_start_time
    print("\nOverall Summary:")
    print("======================================================================")
    print(f"Problem type: {args.problem}")
    print(f"Folder path:  {folder_path}")
    print(f"Total instances processed: {len(results)}")
    print(f"Total processing time (wall-clock): {total_time:.2f} seconds")

    print("\n[FINAL TIME REPORT]")
    print(f"Total training time across all instances: {total_training_time_all:.4f} seconds")
    print(f"Total mapping time across all instances:  {total_mapping_time_all:.4f} seconds")
    print(f"Total post-processing time across all instances: {total_post_time_all:.4f} seconds")
    print(f"TOTAL EVERYTHING (wall-clock):            {total_time:.4f} seconds\n")


if __name__ == '__main__':
    main()
