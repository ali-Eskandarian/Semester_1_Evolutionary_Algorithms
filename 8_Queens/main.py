import numpy as np
from utils import *
from solver import genetic_solver


def main(runs = 50, full_exp = False):

    if full_exp:
        survival_methods = ['Elitism' , 'Generational']
        queens_list = [8,10,12,20]
        # queens_list = [8]
        mutation_probs = [0.01,0.05,0.2,0.5]
        recomb_probs = [0.2, 0.4, 0.6, 0.8]
        recomb_probs = [0.8]
        opt_gen_lists = []  # To store all opt_gen_list for each p_m
        averages_sur = []
        opt_gen_dict = {}
        maxes_sur = []
        for num_q in queens_list:

            # for survive_method in survival_methods:

            # for p_m in mutation_probs:
            # for p_c in recomb_probs:

            p_m, p_c , survive_method = 0.5, 0.8,  'Generational'
            print("=================================")
            print(f"p_m: {p_m}")
            print(f"p_c: {p_c}")
            print(f"num_q: {num_q}")
            print(f"survive_method: {survive_method}")
            avg_fit_list = []
            opt_gen_list = []

            for i in range(runs):
                # print(f"run no {i+1}")
                # solver = genetic_solver(100, num_q, survival_method=survive_method, p_c=0.8, p_m=p_m)
                solver = genetic_solver(100, num_q, survival_method=survive_method, p_c=p_c, p_m=p_m)
                res_opt_gen = solver.evolve(max_iter=10_000)
                avg_fit_list.append(solver.fitnesses)
                opt_gen_list.append(res_opt_gen)

            print(f"Average results: {np.mean(opt_gen_list)}")
            print(f"Max results: {np.max(opt_gen_list)}")
            averages_sur.append(np.mean(opt_gen_list))
            maxes_sur.append(np.max(opt_gen_list))
            opt_gen_lists.append(opt_gen_list)
            opt_gen_dict[survive_method] = opt_gen_list
            plot_all_results(avg_fit_list, runs, survive_method + f'_mutation_prob({p_m})_num_queen({num_q})')
            plot_average_and_variance(avg_fit_list, 1_000, runs,
                                      survive_method + f'_mutation_prob({p_m})_num_queen({num_q})')

            # Store the current opt_gen_list for later plotting

            # After processing all mutation probabilities for this survival method
            # plot_survival_methods_histograms(opt_gen_dict, num_q)
            # plot_multiple_histograms(opt_gen_lists, recomb_probs, survive_method, num_q)
            plot_nqueens_histograms

    else:
        avg_fit_list = []
        opt_gen_list = []
        survive_method = "Elitism" #  Choose either 'Elitism' or 'Generational'
        num_q, p_m = 8 , 0.08
        for i in range(runs):
            solver = genetic_solver(100, num_q, survival_method=survive_method, p_c=0.8, p_m=p_m)
            res_opt_gen = solver.evolve(max_iter=10_000)
            avg_fit_list.append(solver.fitnesses)
            opt_gen_list.append(res_opt_gen)

        print(f"Average results: {np.mean(opt_gen_list)}")
        print(f"max results: {np.max(opt_gen_list)}")

        plot_all_results(avg_fit_list, runs, survive_method + f'_mutation_prob({p_m})_num_queen({num_q})')
        plot_average_and_variance(avg_fit_list, 1_000, runs,
                                  survive_method + f'_mutation_prob({p_m})_num_queen({num_q})')
        plot_histogram(opt_gen_list, runs, survive_method + f'_mutation_prob({p_m})_num_queen({num_q})')

if __name__ == '__main__':

    main(runs = 100, full_exp=True)
