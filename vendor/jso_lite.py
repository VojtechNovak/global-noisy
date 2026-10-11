import numpy as np

class JSO:
    """
    jSO Optimizer (CEC 2017 Winner).
    A specialized variant of L-SHADE with weighted mutation and dynamic p-best selection.
    """
    def __init__(self, dim, bounds, max_evals, seed=None):
        """
        Args:
            dim (int): Dimensionality of the problem.
            bounds (tuple): (lower_bound, upper_bound). Assumes a hypercube.
            max_evals (int): Maximum number of function evaluations.
            seed (int): Random seed for reproducibility.
        """
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        
        # Internal tracking
        self.history = []
        self.eval_count = 0
        self.global_best_fitness = float('inf')
        self.global_best_solution = None

    def run(self, f):
        """
        Executes the optimization loop.
        
        Args:
            f (callable): The objective function to minimize.
            
        Returns:
            dict: Results containing 'fitness_history', 'best_fitness', 'best_solution', 'evals'
        """
        # --- Configuration ---
        # Initial population size (jSO specific logic)
        pop_size_init = int(25 * np.sqrt(self.dim) * np.log10(self.dim))
        if pop_size_init < 30: pop_size_init = 30
        
        pop_size_min = 4
        H = 5  # Memory Size
        
        # --- Initialization ---
        pop = self.rng.uniform(self.bounds[0], self.bounds[1], size=(pop_size_init, self.dim))
        
        # Evaluate initial population
        fitness = np.array([f(x) for x in pop])
        self.eval_count = len(pop)
        
        # Update global bests
        best_idx = np.argmin(fitness)
        self.global_best_fitness = fitness[best_idx]
        self.global_best_solution = pop[best_idx]
        self.history.append(self.global_best_fitness)

        # Memory for F and CR
        M_F = np.ones(H) * 0.5
        M_CR = np.ones(H) * 0.8
        k_mem = 0
        
        # Archive (stores inferior solutions to maintain diversity)
        archive = []
        archive_max_size = pop_size_init
        
        # --- Main Loop ---
        while self.eval_count < self.max_evals:
            # 1. Linear Population Size Reduction (LPSR)
            progress = self.eval_count / self.max_evals
            new_pop_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            current_pop_size = len(pop)
            
            if current_pop_size > new_pop_size:
                n_remove = current_pop_size - new_pop_size
                sort_idx = np.argsort(fitness)
                # Keep best, remove worst
                keep_idx = sort_idx[:new_pop_size]
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                current_pop_size = new_pop_size
                archive_max_size = new_pop_size # Shrink archive capacity

            # 2. Dynamic 'p' for p-best selection
            # Scales from 0.25 (exploration) down to 0.125 (exploitation)
            p_val = 0.25 - (0.25 - 0.125) * progress
            p_best_count = max(2, int(round(p_val * current_pop_size)))

            # 3. Generate adaptive parameters F and CR
            idx_r = self.rng.randint(0, H, size=current_pop_size)
            
            # F generation (Cauchy)
            m_f_vals = M_F[idx_r]
            F = m_f_vals + 0.1 * self.rng.standard_cauchy(size=current_pop_size)
            
            # Retry logic for bad F values
            bad_f = (F <= 0)
            while np.any(bad_f):
                F[bad_f] = m_f_vals[bad_f] + 0.1 * self.rng.standard_cauchy(size=np.sum(bad_f))
                bad_f = (F <= 0)
            F = np.minimum(F, 1.0)
            
            # CR generation (Normal)
            m_cr_vals = M_CR[idx_r]
            CR = self.rng.normal(m_cr_vals, 0.1)
            CR = np.clip(CR, 0.0, 1.0)

            # 4. Mutation & Crossover
            sorted_indices = np.argsort(fitness)
            trials = np.zeros_like(pop)
            
            if len(archive) > 0:
                pop_archive = np.vstack((pop, np.array(archive)))
            else:
                pop_archive = pop

            for i in range(current_pop_size):
                # Select p-best
                pbest_idx = self.rng.choice(sorted_indices[:p_best_count])
                x_pbest = pop[pbest_idx]
                
                # Select r1
                r1 = self.rng.randint(0, current_pop_size)
                while r1 == i:
                    r1 = self.rng.randint(0, current_pop_size)
                x_r1 = pop[r1]
                
                # Select r2
                r2 = self.rng.randint(0, len(pop_archive))
                while r2 == i or r2 == r1:
                    r2 = self.rng.randint(0, len(pop_archive))
                x_r2 = pop_archive[r2]
                
                # Mutation (current-to-pbest-w/1)
                step1 = x_pbest - pop[i]
                step2 = x_r1 - x_r2
                mutant = pop[i] + F[i] * step1 + F[i] * step2
                
                # Crossover
                j_rand = self.rng.randint(0, self.dim)
                cross_mask = self.rng.rand(self.dim) < CR[i]
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Boundary Handling (Clipping for stability)
                trials[i] = np.clip(trial, self.bounds[0], self.bounds[1])

            # 5. Selection
            success_mem_F = []
            success_mem_CR = []
            diff_fitness = []
            
            for i in range(current_pop_size):
                if self.eval_count >= self.max_evals: break
                
                f_trial = f(trials[i])
                self.eval_count += 1
                self.history.append(self.global_best_fitness) # Log per eval or per generation
                
                # Update global best immediately if found
                if f_trial < self.global_best_fitness:
                    self.global_best_fitness = f_trial
                    self.global_best_solution = trials[i]

                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        success_mem_F.append(F[i])
                        success_mem_CR.append(CR[i])
                        diff_fitness.append(fitness[i] - f_trial)
                    
                    pop[i] = trials[i]
                    fitness[i] = f_trial

            # Maintain Archive Size
            if len(archive) > archive_max_size:
                n_excess = len(archive) - archive_max_size
                # Random removal
                remove_indices = self.rng.choice(len(archive), n_excess, replace=False)
                archive = [x for idx, x in enumerate(archive) if idx not in remove_indices]

            # 6. Update Memory (Weighted Lehman Mean)
            if len(success_mem_F) > 0:
                diff_fitness = np.array(diff_fitness)
                total_diff = np.sum(diff_fitness)
                
                if total_diff > 0:
                    weights = diff_fitness / total_diff
                    
                    denom = np.sum(weights * np.array(success_mem_F))
                    if denom > 0:
                        mean_f = np.sum(weights * (np.array(success_mem_F)**2)) / denom
                    else:
                        mean_f = M_F[k_mem]

                    mean_cr = np.sum(weights * np.array(success_mem_CR))
                    
                    M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                    
                    k_mem = (k_mem + 1) % H
                    M_CR = np.clip(M_CR, 0, 1)
                    M_F = np.clip(M_F, 0, 1)
        
        return {
            "fitness_history": np.array(self.history),
            "best_fitness": self.global_best_fitness,
            "best_solution": self.global_best_solution,
            "evals": self.eval_count
        }