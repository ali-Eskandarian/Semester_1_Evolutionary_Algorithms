from utils import *
import numpy as np

class genetic_solver:
    def __init__(self, num_chr, num_queen=8, survival_method="Elitism", p_c=0.8, p_m=0.05):
        self.num_chr             = num_chr          # Population size
        self.p_c                 = p_c              # probability of crossover leads to offsprings
        self.p_m                 = p_m              # probability of mutation
        self.survival_method     = survival_method
        if self.survival_method not in ["Elitism", "Generational"]:
            raise ValueError("Invalid survival_method. Choose either 'Elitism' or 'Generational'.")
        self.num_queen           = num_queen
        self.pop                 = {}
        self.scores              = {}
        self.fitnesses           = [0]
        self.fitness_max         = 0
        self.fitness_avg         = 0

    def _initialize(self):
        for i in range(self.num_chr):
            chrom               = np.random.permutation(self.num_queen)
            score               = self._fitness_score(chrom)
            self.pop   [f'{i}'] = chrom
            self.scores[f'{i}'] = score

        self._store_results()

    def _store_results(self):
        self.fitness_max = max(self.scores.values())
        self.fitness_avg = np.mean(list(self.scores.values()))
        self.fitnesses.append(self.fitness_avg)

    def _fitness_score(self, seq):
        score = 0

        for row in range(self.num_queen):
            col = seq[row]
            for other_row in range(self.num_queen):
                # queens cannot pair with itself , To clear the Logic
                if other_row == row:
                    continue
                if seq[other_row] == col:
                    continue
                if other_row + seq[other_row] == row + col:
                    continue
                if other_row - seq[other_row] == row - col:
                    continue
                # if every pair of queens are non-attacking.
                score += 1

        # divide by 2 as pairs of queens are commutative
        return score / 2

    def _plot(self):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.fitnesses)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Fitness Progress (8-Queens)")
        fig.savefig("plots/results.png")
        plt.close()

    def _mating(self, mating_pool):
        mating_pool      = np.array([random.choice(mating_pool) for _ in range(5)])
        selected_parents = [self.scores[f'{k}'] for k in mating_pool]
        parents          = []

        for _ in range(2):
            max_f = max(selected_parents)
            parents.append(selected_parents.index(max_f))
            selected_parents.pop(selected_parents.index(max_f))
        return self.pop[f'{parents[0]}'], self.pop[f'{parents[1]}']

    def _crossover(self, final_parents):
        parent_1, parent_2 = final_parents

        if np.random.random()<self.p_c:
            pos_recombination = np.random.randint(1,self.num_queen)
            part_a_1, part_b_1 = parent_1[:pos_recombination], parent_1[pos_recombination:]
            part_a_2, part_b_2 = parent_2[:pos_recombination], parent_2[pos_recombination:]
            child_1            = np.array([*Counter(np.concatenate([part_a_1, part_b_2, part_a_2]))])
            child_2            = np.array([*Counter(np.concatenate([part_a_2, part_b_1, part_a_1]))])
            return child_1, child_2

        else:
            return final_parents

    def _mutation(self, created_children):
        mutated_children = [0,0]
        for idx, child in enumerate(created_children):
            if np.random.random() < self.p_m:
                swap_poses                                 = np.random.randint(0,self.num_queen, size = 2)
                child[swap_poses[0]], child[swap_poses[1]] =  child[swap_poses[1]], child[swap_poses[0]]
                mutated_children[idx]                      = child
            else:
                mutated_children[idx] = child

        return mutated_children

    def _survival_selection(self, final_children):
        if self.survival_method=='Elitism':
            score_1, score_2 = self._fitness_score(final_children[0]), self._fitness_score(final_children[1])
            if score_1 < score_2:
              final_children   = final_children[::-1]
              score_1, score_2 = score_2, score_1
            child_scores = (score_1, score_2)
            all_values   = list(self.scores.values())
            idx_loop     = 0

            for _ in range(self.num_chr):
                max_value = max(all_values)
                ind       = all_values.index(max_value)
                if child_scores[idx_loop] > max_value > 0:
                    self.pop[f'{ind}']    = final_children[idx_loop]
                    self.scores[f'{ind}'] = child_scores[idx_loop]

                    idx_loop += 1
                    if idx_loop == 2:
                        break

                all_values[ind] = -1

            self._store_results()

        elif self.survival_method=='Generational':
            for i in range(self.num_chr):
                self.pop[f'{i}']    = final_children[i]
                self.scores[f'{i}'] = self._fitness_score(final_children[i])

            self._store_results()

    def evolve(self, max_iter):
        self._initialize()
        generation_num = 1

        while self.fitness_max < ((self.num_queen * (self.num_queen-1))/ 2) and generation_num < max_iter:
            if self.survival_method=='Elitism':
                mating_pool    = np.array([random.choice(range(self.num_chr)) for _ in range(self.num_chr)])
                final_parents  = self._mating(mating_pool)
                offsprings     = self._crossover(final_parents)
                final_children = self._mutation(offsprings)
                self._survival_selection(final_children)

            elif self.survival_method=='Generational':
                final_children_list = []
                for ind in range(int(self.num_chr/2)):
                    mating_pool    = np.array([random.choice(range(self.num_chr)) for _ in range(self.num_chr)])
                    final_parents  = self._mating(mating_pool)
                    offsprings     = self._crossover(final_parents)
                    final_children = self._mutation(offsprings)
                    final_children_list.append(final_children[0])
                    final_children_list.append(final_children[1])
                self._survival_selection(final_children_list)
            generation_num += 1

        self._plot()
        return generation_num