import torch
import math
import time

from src.loss import combined_loss
from src.utils import (
    batch_coverage_check,
    optimize_solution_size,
    integrated_post_process_set_cover,
    integrated_post_process_hitting_set,
    post_process_solution_hypermaxcut,
    post_process_solution_hypermultiwaycut,
    post_process_solution_subset_sum
)


def dynamic_threshold(probs: torch.Tensor) -> float:
    """
    Compute a dynamic threshold = mean(prob) + 0.5*std(prob),
    clamped between 0.05 and 0.95.
    """
    mean_p = probs.mean().item()
    std_p = probs.std().item()
    thr = mean_p + 0.5 * std_p
    thr = max(0.05, min(thr, 0.95))
    return thr


def get_training_coverage(probs: torch.Tensor,
                          incidence_matrix: torch.Tensor,
                          problem_type: str = "set_cover",
                          extra_info=None):
    """
    Evaluate coverage (for set_cover/hitting_set) or fraction cut, etc., used in training.
    """
    device = probs.device

    if problem_type == "set_cover":
        thr = dynamic_threshold(probs)
        raw_sol = (probs >= thr).float()
        coverage_ratio, uncovered_count = batch_coverage_check(raw_sol, incidence_matrix)
        return coverage_ratio, uncovered_count

    elif problem_type == "hitting_set":
        thr = dynamic_threshold(probs)
        raw_sol = (probs >= thr).float()
        if incidence_matrix.is_sparse:
            coverage_vals = torch.sparse.mm(incidence_matrix, raw_sol.unsqueeze(1)).squeeze(1)
        else:
            coverage_vals = incidence_matrix @ raw_sol.unsqueeze(1)
            coverage_vals = coverage_vals.squeeze(1)
        covered_count = (coverage_vals >= 1.0).sum().item()
        total_sets = incidence_matrix.size(0)
        ratio = covered_count / float(total_sets) if total_sets > 0 else 1.0
        uncovered_count = total_sets - covered_count
        return ratio, uncovered_count

    elif problem_type == "subset_sum":
        if extra_info is None:
            return 0.0, 0
        (weights, target_sum) = extra_info
        w_t = torch.tensor(weights, device=device)
        sum_ = (probs * w_t).sum().item()
        ratio = sum_ / (target_sum + 1e-9)
        return ratio, 0

    elif problem_type == "hypermaxcut":
        n_edges = incidence_matrix.size(1)
        cut_count = 0
        for j in range(n_edges):
            if incidence_matrix.is_sparse:
                mask = (incidence_matrix.indices()[1] == j)
                node_idx = incidence_matrix.indices()[0][mask]
            else:
                col_j = incidence_matrix[:, j]
                node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
            if len(node_idx) == 0:
                continue
            vals = probs[node_idx]
            if (vals.min() < 0.5) and (vals.max() > 0.5):
                cut_count += 1
        ratio = cut_count / (n_edges + 1e-9)
        return ratio, 0

    elif problem_type == "hypermultiwaycut":
        n_edges = incidence_matrix.size(1)
        assignments = torch.argmax(probs, dim=1)
        cut_count = 0
        for j in range(n_edges):
            if incidence_matrix.is_sparse:
                mask = (incidence_matrix.indices()[1] == j)
                node_idx = incidence_matrix.indices()[0][mask]
            else:
                col_j = incidence_matrix[:, j]
                node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
            if len(node_idx) == 0:
                continue
            parts_used = torch.unique(assignments[node_idx])
            if len(parts_used) > 1:
                cut_count += 1
        ratio = cut_count / (n_edges + 1e-9)
        return ratio, 0

    else:
        return 0.0, 0


def coverage_injection(model: torch.nn.Module, scale: float = 0.05):
    """
    Inject random noise into model parameters in-place to 'unstick' from local minima.
    """
    for param in model.parameters():
        param.data.add_(scale * torch.randn_like(param))


def reinitialize_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, meltdown_times: int):
    """
    Reinitialize model parameters in-place.
    If meltdown_times>1, add extra noise. Clears optimizer state.
    """
    for param in model.parameters():
        param.data.uniform_(-0.01, 0.01)
        if meltdown_times > 1:
            param.data.add_(0.05 * torch.randn_like(param))
    optimizer.state.clear()


def get_final_raw_solution(probs: torch.Tensor,
                           incidence_matrix: torch.Tensor,
                           problem_type="set_cover",
                           extra_info=None):
    """
    Return the "raw" solution from the neural net.

    For set_cover/hitting_set => pure threshold approach (no integrated or local improvement).
    For subset_sum => your original approach with threshold comparison.
    For hypermaxcut/hypermultiwaycut => your original approach with local flipping inside.
    """
    import time
    from src.utils import (
        batch_coverage_check,
        optimize_solution_size,
        integrated_post_process_set_cover,
        integrated_post_process_hitting_set,
        post_process_solution_hypermaxcut_fast,
        post_process_solution_hypermultiwaycut,
        post_process_solution_subset_sum
    )
    device = probs.device
    raw_post_time = 0.0

    if problem_type == "set_cover":
        thr_dyn = dynamic_threshold(probs)
        sol_dyn = (probs >= thr_dyn).float()
        cov_dyn, uncov_dyn = batch_coverage_check(sol_dyn, incidence_matrix)

        sol_05 = (probs >= 0.5).float()
        cov_05, uncov_05 = batch_coverage_check(sol_05, incidence_matrix)

        if cov_dyn > cov_05:
            chosen_sol = sol_dyn
            chosen_cov = cov_dyn
            chosen_unc = uncov_dyn
            chosen_thr = thr_dyn
        elif abs(cov_05 - cov_dyn) < 1e-9:
            if sol_05.sum() < sol_dyn.sum():
                chosen_sol = sol_05
                chosen_cov = cov_05
                chosen_unc = uncov_05
                chosen_thr = 0.5
            else:
                chosen_sol = sol_dyn
                chosen_cov = cov_dyn
                chosen_unc = uncov_dyn
                chosen_thr = thr_dyn
        else:
            chosen_sol = sol_05
            chosen_cov = cov_05
            chosen_unc = uncov_05
            chosen_thr = 0.5

        return {
            'solution': chosen_sol,
            'coverage_ratio': chosen_cov,
            'valid': (chosen_unc == 0),
            'size': float(chosen_sol.sum().item()),
            'mean_prob': float(probs.mean().item()),
            'std_prob': float(probs.std().item()),
            'dynamic_thr_value': thr_dyn,
            'threshold_05_value': 0.5,
            'raw_post_time': raw_post_time
        }

    elif problem_type == "hitting_set":
        thr_dyn = dynamic_threshold(probs)
        sol_dyn = (probs >= thr_dyn).float()
        if incidence_matrix.is_sparse:
            coverage_vals_dyn = torch.sparse.mm(incidence_matrix, sol_dyn.unsqueeze(1)).squeeze(1)
        else:
            coverage_vals_dyn = incidence_matrix @ sol_dyn.unsqueeze(1)
            coverage_vals_dyn = coverage_vals_dyn.squeeze(1)
        covered_dyn = (coverage_vals_dyn >= 1.0).sum().item()
        total_sets = incidence_matrix.size(0)
        ratio_dyn = covered_dyn / (total_sets + 1e-9)

        sol_05 = (probs >= 0.5).float()
        if incidence_matrix.is_sparse:
            coverage_vals_05 = torch.sparse.mm(incidence_matrix, sol_05.unsqueeze(1)).squeeze(1)
        else:
            coverage_vals_05 = incidence_matrix @ sol_05.unsqueeze(1)
            coverage_vals_05 = coverage_vals_05.squeeze(1)
        covered_05 = (coverage_vals_05 >= 1.0).sum().item()
        ratio_05 = covered_05 / (total_sets + 1e-9)

        if ratio_dyn > ratio_05:
            final_sol = sol_dyn
            final_cov = ratio_dyn
        elif abs(ratio_05 - ratio_dyn) < 1e-9:
            if sol_05.sum() < sol_dyn.sum():
                final_sol = sol_05
                final_cov = ratio_05
            else:
                final_sol = sol_dyn
                final_cov = ratio_dyn
        else:
            final_sol = sol_05
            final_cov = ratio_05

        return {
            'solution': final_sol,
            'coverage_ratio': final_cov,
            'valid': (final_cov >= 0.995),
            'size': float(final_sol.sum().item()),
            'mean_prob': float(probs.mean().item()),
            'std_prob': float(probs.std().item()),
            'raw_post_time': 0.0
        }

    elif problem_type == "subset_sum":
        thr_dyn = dynamic_threshold(probs)
        sol_dyn = (probs >= thr_dyn).float()
        sol_05 = (probs >= 0.5).float()

        if extra_info is None:
            (weights, target_sum) = ([], 1.0)
        else:
            (weights, target_sum) = extra_info

        w_t = torch.tensor(weights, device=device)
        sum_dyn = float((sol_dyn * w_t).sum().item())
        sum_05 = float((sol_05 * w_t).sum().item())
        ratio_dyn = sum_dyn / (target_sum + 1e-9)
        ratio_05 = sum_05 / (target_sum + 1e-9)

        dist_dyn = abs(ratio_dyn - 1.0)
        dist_05 = abs(ratio_05 - 1.0)
        if dist_dyn <= dist_05:
            chosen_sol = sol_dyn
            chosen_ratio = ratio_dyn
            chosen_thr = thr_dyn
        else:
            chosen_sol = sol_05
            chosen_ratio = ratio_05
            chosen_thr = 0.5

        return {
            'solution': chosen_sol,
            'threshold': chosen_thr,
            'coverage_ratio': chosen_ratio,
            'valid': True,
            'size': float(chosen_sol.sum().item()),
            'mean_prob': float(probs.mean().item()),
            'std_prob': float(probs.std().item()),
            'raw_post_time': 0.0
        }

    elif problem_type == "hypermaxcut":
        from src.utils import post_process_solution_hypermaxcut_fast

        thr_dyn = dynamic_threshold(probs)
        sol_dyn = (probs >= thr_dyn).float()
        sol_05 = (probs >= 0.5).float()

        n_edges = incidence_matrix.size(1)

        def frac_cut(sv):
            ccount = 0
            for j in range(n_edges):
                if incidence_matrix.is_sparse:
                    mask = (incidence_matrix.indices()[1] == j)
                    node_idx = incidence_matrix.indices()[0][mask]
                else:
                    col_j = incidence_matrix[:, j]
                    node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
                if len(node_idx) == 0:
                    continue
                vals = sv[node_idx]
                if (vals.min() < 0.5) and (vals.max() > 0.5):
                    ccount += 1
            return ccount / (n_edges + 1e-9)

        ratio_dyn = frac_cut(sol_dyn)
        ratio_05 = frac_cut(sol_05)

        if ratio_dyn >= ratio_05:
            chosen_sol = sol_dyn
            chosen_ratio = ratio_dyn
            chosen_thr = thr_dyn
        else:
            chosen_sol = sol_05
            chosen_ratio = ratio_05
            chosen_thr = 0.5

        raw_post_time = 0.0
        if chosen_ratio < 0.995:
            t2 = time.time()
            improved_sol = post_process_solution_hypermaxcut_fast(chosen_sol, incidence_matrix, max_passes=2)
            local_time = time.time() - t2
            raw_post_time += local_time
            ratio_imp = frac_cut(improved_sol)
            if ratio_imp > chosen_ratio:
                chosen_sol = improved_sol
                chosen_ratio = ratio_imp

        return {
            'solution': chosen_sol,
            'threshold': chosen_thr,
            'coverage_ratio': chosen_ratio,
            'valid': True,
            'size': float((chosen_sol >= 0.5).sum().item()),
            'mean_prob': float(probs.mean().item()),
            'std_prob': float(probs.std().item()),
            'which_threshold_was_used': (
                "dynamic_threshold" if ratio_dyn >= ratio_05 else "fixed_threshold_0.5"
            ),
            'raw_chosen_coverage_ratio': max(ratio_dyn, ratio_05),
            'solution_dyn_raw': sol_dyn,
            'coverage_ratio_dyn_raw': ratio_dyn,
            'solution_05_raw': sol_05,
            'coverage_ratio_05_raw': ratio_05,
            'dynamic_thr_value': thr_dyn,
            'threshold_05_value': 0.5,
            'raw_post_time': raw_post_time
        }

    elif problem_type == "hypermultiwaycut":
        from src.utils import post_process_solution_hypermultiwaycut

        n_nodes = probs.size(0)
        n_edges = incidence_matrix.size(1)
        assignments = torch.argmax(probs, dim=1)

        cut_count_main = 0
        for j in range(n_edges):
            if incidence_matrix.is_sparse:
                mask = (incidence_matrix.indices()[1] == j)
                node_idx = incidence_matrix.indices()[0][mask]
            else:
                col_j = incidence_matrix[:, j]
                node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
            if len(node_idx) == 0:
                continue
            used_parts = torch.unique(assignments[node_idx])
            if len(used_parts) > 1:
                cut_count_main += 1
        ratio_main = cut_count_main / (n_edges + 1e-9)

        threshold_05_assign = []
        for i in range(n_nodes):
            row = probs[i]
            valid_parts = (row >= 0.5).nonzero(as_tuple=True)[0]
            if len(valid_parts) > 0:
                best_p = valid_parts[row[valid_parts].argmax().item()].item()
            else:
                best_p = row.argmax().item()
            threshold_05_assign.append(best_p)
        threshold_05_assign = torch.tensor(threshold_05_assign, device=probs.device)

        cut_count_05 = 0
        for j in range(n_edges):
            if incidence_matrix.is_sparse:
                mask = (incidence_matrix.indices()[1] == j)
                node_idx = incidence_matrix.indices()[0][mask]
            else:
                col_j = incidence_matrix[:, j]
                node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
            if len(node_idx) == 0:
                continue
            used_p_05 = torch.unique(threshold_05_assign[node_idx])
            if len(used_p_05) > 1:
                cut_count_05 += 1
        ratio_05 = cut_count_05 / (n_edges + 1e-9)

        if ratio_main >= ratio_05:
            chosen_label = "argmax_dynamic"
            raw_ratio = ratio_main
            chosen_sol = assignments
        else:
            chosen_label = "threshold_0.5"
            raw_ratio = ratio_05
            chosen_sol = threshold_05_assign

        raw_post_time = 0.0
        final_assign = chosen_sol.clone()
        if raw_ratio < 0.95:
            t0 = time.time()
            new_assign = post_process_solution_hypermultiwaycut(probs, incidence_matrix, max_passes=2)
            local_time = time.time() - t0
            raw_post_time += local_time

            cut_count2 = 0
            for j in range(n_edges):
                if incidence_matrix.is_sparse:
                    mask = (incidence_matrix.indices()[1] == j)
                    node_idx = incidence_matrix.indices()[0][mask]
                else:
                    col_j = incidence_matrix[:, j]
                    node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
                if len(node_idx) == 0:
                    continue
                used_p2 = torch.unique(new_assign[node_idx])
                if len(used_p2) > 1:
                    cut_count2 += 1
            ratio2 = cut_count2 / (n_edges + 1e-9)
            if ratio2 > raw_ratio:
                raw_ratio = ratio2
                final_assign = new_assign

        return {
            'solution': final_assign,
            'coverage_ratio': raw_ratio,
            'valid': True,
            'size': float(n_nodes),
            'mean_prob': float(probs.mean().item()),
            'std_prob': float(probs.std().item()),
            'which_raw_solution_used': chosen_label,
            'raw_chosen_coverage_ratio': max(ratio_main, ratio_05),
            'solution_dyn_raw': assignments,
            'coverage_ratio_dyn_raw': ratio_main,
            'solution_05_raw': threshold_05_assign,
            'coverage_ratio_05_raw': ratio_05,
            'raw_post_time': raw_post_time
        }

    else:
        return {
            'solution': probs,
            'coverage_ratio': 0.0,
            'valid': True,
            'size': float(probs.size(0)),
            'raw_post_time': 0.0
        }


def train_model(model: torch.nn.Module,
                incidence_matrix: torch.Tensor,
                params: dict,
                problem_type: str = "set_cover",
                extra_info=None) -> torch.Tensor:
    """
    Main training loop with unified adaptive restart mechanism.
    *Uses unified meltdown threshold of 0.05 for all problem types
    *Adds special early stopping rules for large instances:
      - >=10000 rows: if quality >=1.90 => early stop
      - >=30000 rows: if quality >=1.85 => early stop
    *For hypermaxcut/hypermultiwaycut: if #nodes<1000 => stop if coverage >=0.9,
      else coverage >=0.99. 
    """
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = incidence_matrix.device
    n_rows = incidence_matrix.size(0)

    meltdown_floor = 0.05
    meltdown_cov_threshold = 0.05
    
    meltdown_limit = 10
    meltdown_times = 0
    meltdown_streak = 0
    noise_inject_streak = 0

    local_max_epochs = params.get('max_epochs', 200)
    local_patience = params.get('patience', 10)
    local_lr = params.get('lr', 0.0005)

    extremely_large_instance = False
    large_instance = False
    if problem_type in ["set_cover", "hitting_set"]:
        if n_rows >= 20000:
            extremely_large_instance = True
        elif n_rows >= 10000:
            large_instance = True

    if extremely_large_instance:
        local_max_epochs = 10
    elif large_instance:
        local_max_epochs = min(local_max_epochs, 50)
        local_patience = min(local_patience, 10)

    coverage_floor = params.get('coverage_floor', 0.4)
    coverage_stall_count = 0
    coverage_floor_patience = params.get('coverage_floor_patience', 15)

    if problem_type == "set_cover":
        phase2_cov = params.get('phase2_cov', 0.85)
    else:
        phase2_cov = params.get('phase2_cov', 0.98)
    phase2_patience = params.get('phase2_patience', 3)
    phase2_count = 0
    in_phase2 = False

    wd = params.get('weight_decay', 0.0005)
    optimizer = Adam(model.parameters(), lr=local_lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=local_max_epochs, eta_min=local_lr * 0.05)

    best_probs = None
    best_quality = -999999.0
    best_epoch = 0
    no_improve = 0
    best_cov_so_far = 0.0
    prev_cov = 0.0

    for epoch in range(1, local_max_epochs + 1):
        model.train()
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch, local_max_epochs)
        
        optimizer.zero_grad()

        probs_init = model(incidence_matrix)
        if problem_type not in ["hypermultiwaycut"]:
            probs_init = probs_init.clamp(1e-3, 1-1e-3)

        cov_ratio, uncovered = get_training_coverage(
            probs_init, incidence_matrix,
            problem_type=problem_type,
            extra_info=extra_info
        )

        if cov_ratio > best_cov_so_far:
            best_cov_so_far = cov_ratio

        meltdown_done = False
        if cov_ratio < meltdown_floor:
            meltdown_streak += 1
        else:
            meltdown_streak = 0

        if cov_ratio < coverage_floor:
            noise_inject_streak += 1
        else:
            noise_inject_streak = 0

        can_meltdown = (best_cov_so_far < meltdown_cov_threshold)
        if (meltdown_streak >= meltdown_limit
            and meltdown_times < 2
            and (epoch + meltdown_limit < local_max_epochs)
            and can_meltdown):
            meltdown_times += 1
            meltdown_streak = 0
            print(f"\n[!] Detected meltdown. meltdown_times={meltdown_times}")
            reinitialize_model(model, optimizer, meltdown_times)
            p_ri = model(incidence_matrix)
            if problem_type not in ["hypermultiwaycut"]:
                p_ri = p_ri.clamp(1e-3, 1-1e-3)
            c_ri, _ = get_training_coverage(p_ri, incidence_matrix, problem_type, extra_info)
            if c_ri > best_cov_so_far:
                best_cov_so_far = c_ri
            meltdown_done = True
            probs = p_ri
        else:
            meltdown_done = False
            meltdown_exhausted_or_disabled = (meltdown_times >= 2) or (not can_meltdown)
            if (meltdown_exhausted_or_disabled and (cov_ratio < coverage_floor)
                and noise_inject_streak >= meltdown_limit):
                print(f"[!] meltdown not possible => injecting noise. coverage= {cov_ratio:.4f}")
                noise_inject_streak = 0
                coverage_injection(model, scale=0.05)
                p_inj = model(incidence_matrix)
                if problem_type not in ["hypermultiwaycut"]:
                    p_inj = p_inj.clamp(1e-3, 1-1e-3)
                c_inj, _ = get_training_coverage(p_inj, incidence_matrix, problem_type, extra_info)
                if c_inj > best_cov_so_far:
                    best_cov_so_far = c_inj
                probs = p_inj
            else:
                probs = probs_init

        if problem_type == "set_cover":
            n_elems = incidence_matrix.size(0)
            n_subsets = incidence_matrix.size(1)
            
            estimated_optimal_sets = max(math.log(n_elems), math.sqrt(n_elems) / 3)
            ideal_density = estimated_optimal_sets / n_subsets
            ideal_density = max(ideal_density, 0.01)
            
            mean_p = probs.mean().item()
            density_diff = abs(mean_p - ideal_density)
            
            efficiency_bonus = max(0, 1.0 - mean_p / ideal_density) if ideal_density > 0 else 0
            
            quality_score = cov_ratio + (1.0 - density_diff) + efficiency_bonus

        elif problem_type == "hitting_set":
            ideal_density = 0.005
            density_diff = abs(probs.mean().item() - ideal_density)
            quality_score = cov_ratio + (1.0 - density_diff)

        elif problem_type == "subset_sum":
            dist1 = abs(1.0 - cov_ratio)
            quality_score = 2.0 - 10.0 * dist1

        elif problem_type in ["hypermaxcut", "hypermultiwaycut"]:
            quality_score = cov_ratio
        else:
            quality_score = 0.0

        if quality_score > best_quality:
            best_quality = quality_score
            best_probs = probs.detach().clone()
            best_epoch = epoch
            no_improve = 0
            print(f"  [*] New best solution. Quality={best_quality:.4f}\n")
        else:
            no_improve += 1

        if cov_ratio >= phase2_cov:
            phase2_count += 1
        else:
            phase2_count = 0
        if phase2_count >= phase2_patience:
            in_phase2 = True

        if (cov_ratio < coverage_floor) and (prev_cov < 0.8):
            coverage_stall_count += 1
        else:
            coverage_stall_count = 0
        prev_cov = cov_ratio

        if coverage_stall_count >= coverage_floor_patience:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.7
            coverage_stall_count = 0
            print(" [!] Coverage ratio stuck; reducing LR by factor 0.7.\n")

        loss_val = combined_loss(
            probs, incidence_matrix,
            epoch, local_max_epochs,
            in_phase2=in_phase2,
            problem_type=problem_type,
            extra_info=extra_info
        )
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch}/{local_max_epochs}:")
        print(f"Loss: {loss_val.item():.4f}")
        print(f"Coverage (dyn thr): {cov_ratio:.4f}, Uncovered={uncovered}")
        print(f"Prob Dist: mean={probs.mean().item():.4f}, std={probs.std().item():.4f}, "
              f"min={probs.min().item():.3f}, max={probs.max().item():.3f}")
        if problem_type not in ["hypermultiwaycut"]:
            mp_ = (probs * (1 - probs)).mean().item()
            print(f"Mid penalty p*(1-p): {mp_:.4f}")
        if in_phase2:
            print("  [Phase 2 active: smaller-solution penalty]")


        if problem_type in ["set_cover", "hitting_set"]:
            if n_rows >= 30000:
                if quality_score >= 1.85:
                    print(" [!] Very large instance => quality≥1.85 => early stop.")
                    break
            elif n_rows >= 10000:
                if quality_score >= 1.90:
                    print(" [!] Large instance => quality≥1.90 => early stop.")
                    break

        if problem_type == "subset_sum":
            if cov_ratio >= 0.9:
                print("High subset-sum ratio ≥0.9 => early stop.")
                break
        elif problem_type in ["hypermaxcut", "hypermultiwaycut"]:
            n_nodes = incidence_matrix.size(0)
            threshold_coverage = 0.9 if n_nodes < 1000 else 0.99
            if cov_ratio >= threshold_coverage:
                print(f"High coverage/cut ratio ≥{threshold_coverage} => early stop (n_nodes={n_nodes}).")
                break

        if (epoch >= params.get('min_epochs', 30)) and (no_improve >= local_patience):
            print(f"\nEarly stopping triggered. best_epoch={best_epoch}, best_quality={best_quality:.4f}")
            break

    print("Training completed:")
    print(f"Best epoch: {best_epoch}")
    print(f"Final best quality score: {best_quality:.4f}")

    if best_probs is None:
        return torch.zeros(incidence_matrix.size(1), device=device)

    return best_probs

