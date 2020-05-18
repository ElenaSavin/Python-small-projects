from collections import defaultdict
from multiprocessing import Pool
import argparse
from random import sample, choice, randrange
from enum import Enum, auto
from typing import Dict
import numpy as np
from itertools import count, chain
from datetime import datetime
import os
from tkinter import Tk, Canvas, Frame, BOTH
from sys import maxsize as MAXSIZE
from math import ceil


NUM_OF_GENETIC_UNITS = 12


NEIGHBOURS_IDS_LISTS = {1: [2, 6, 11], 2: [3, 7, 8, 9, 11], 3: [4, 5, 9, 11], 4: [5, 11], 5: [9, 11, 12],
                        6: [7, 8, 10, 11, 12], 7: [8], 8: [9, 10], 9: [10, 12], 10: [12], 11: [12], 12: []}


class Game:
    @staticmethod
    def play(out_dir, subprocesses, simulation_viz):
        """
        Play all kinds of different simulations.
        @param out_dir: directory of the output file with simulation stats.
        @param subprocesses: number of subprocesses.
        @param simulation_viz: indicates if the simulation is with a visual window i.e GUI.
        """
        simulator = Simulator()
        if simulation_viz is SimulationViz.FRONT:
            simulator.simulate_front_simulation()
        else:
            simulator.simulate_background_simulations(out_dir, subprocesses)


class Simulator:
    """Play simulation(s), and write statistics and final solutions of background simulations to files."""
    def simulate_front_simulation(self):
        """
        Play a GUI simulation.
        """
        factory = SimulationFactory()
        simulation = factory.create_front_simulation()
        simulation.play()

    def simulate_background_simulations(self, out_dir, subprocesses):
        """
        Run simulations in parallel and write their statistics and final solution to files.

        Get all_simulations_by_groups, a list of lists of simulations, where each inner list's size is the number of
        subprocesses to run in parallel.
        Then run each group of simulations (each inner list in all_simulations_by_groups) in parallel.
        @param out_dir: directory of the output file with simulation stats.
        @param subprocesses: number of subprocesses
        """
        all_output_lines = []
        factory = SimulationFactory(subprocesses)
        all_simulations_by_groups = factory.create_all_background_simulations()
        for group_num, group_of_simulations in enumerate(all_simulations_by_groups):
            self.report_progress(group_num, len(group_of_simulations))
            with Pool() as pool:
                output_lines_of_one_group = pool.map(Simulator.start_background_simulation, group_of_simulations)
                all_output_lines.extend(output_lines_of_one_group)
        self.report_end()
        self.write_background_stats(out_dir, all_output_lines)

    def report_progress(self, group_num, group_length):
        """
        This method reports (prints) the simulation progress to the console.
        @param group_num: number of group.
        @param group_length: group length.
        """
        first_sim_in_group = (group_num * group_length) + 1
        last_sim_in_group = first_sim_in_group + group_length - 1
        if first_sim_in_group < last_sim_in_group:
            print(f"{datetime.now()}\t{first_sim_in_group}-{last_sim_in_group}\n")
        else:
            print(f"{datetime.now()}\t{first_sim_in_group}\n")

    def report_end(self):
        """
        Print the time the simulation ended.
        """
        print(f"\n\n\n{datetime.now()}\tDone\n")

    def write_background_stats(self, out_dir, all_output_lines):
        """
        This method writes the stats to a csv file in the given directory.
        @param out_dir: directory of the output file with simulation stats.
        @param all_output_lines: lines of output containing the stats.
        """
        out_file = os.path.join(out_dir, "simulations_stats.csv")
        with open(out_file, "w") as file:
            col_names = "sim_id,round,best_score,avg_score,solutions,elitist_candidates,elitism_rate,discard_rate," \
                        "mutation_rate,colors\n"
            file.write(col_names)
            for lines_of_one_simulation in all_output_lines:
                for line in lines_of_one_simulation:
                    file.write(line)

    @classmethod
    def start_background_simulation(cls, simulation):
        """
        Used in order to achieve multiprocessing with pool.map.
        @param simulation: The simulation running.
        @return: lines with stats for the given simulation.
        """
        output_lines_of_one_group = simulation.play()
        return output_lines_of_one_group


class SimulationViz(Enum):
    """Denotes the visualization of a simulation.

    For a number of simulations running in parallel, the first one is FRONT and is printed (drawn) to the screen, and
    the other ones are BACK (because they are run and calculated without graphical representation)."""
    FRONT = auto()
    BACK = auto()


class SimulationFactory:
    """This is a simulation factory class. It creates all kinds of simulations."""
    def __init__(self, subprocesses=1):
        """
        @param subprocesses: number of subprocesses
        """
        self.subprocesses = subprocesses
        self.sorting_order = ScoresSortingOrder.ASCENDING
        self.chance_zones = 10
        self.first_breeding_zone_chance = 0.7
        self.last_breeding_zone_chance = 0.3
        self.fitting_function = FittingFunction(self.sorting_order)
        self.crossover_maker = CrossoverMaker()
        self.neighbours_ids_lists = NEIGHBOURS_IDS_LISTS  # other maps with different polygons could be used
        self.all_num_of_solutions = (30, 100, 150)
        self.all_elitist_candidates = (1, 2, 3)
        self.all_elitism_rates = (0.05, 0.1)
        self.all_discard_rates = (0.05, 0.1)
        self.all_mutation_rates = (0.005, 0.01)
        self.num_of_revisions = 3

    def create_breading_rules(self, elitist_candidates, elitism_rate, discard_rate):
        """
        @param elitist_candidates: number of candidates that will be addressed and managed as elitists.
        @param elitism_rate: final percentage of elitists in a population
        @param discard_rate: percentage of population to discard.
        @return: a set of rules for breeding.
        """
        return BreedingRules(self.sorting_order, elitist_candidates, elitism_rate, discard_rate,
                             self.chance_zones, self.first_breeding_zone_chance, self.last_breeding_zone_chance)

    def create_mutation_rules(self, mutation_rate):
        """
        @param mutation_rate: percentage of mutation.
        @return: mutation rules.
        """
        return MutationRules(mutation_rate)

    def create_mutation_maker(self, mutation_rules):
        """
        @param mutation_rules: percentage of mutation.
        @return: mutation maker with the above mutation rules.
        """
        return MutationMaker(mutation_rules)

    def create_breeder(self, breeding_rules):
        """
        @param breeding_rules: a set of rules for breeding.
        @return: a breeder which will breed according to there rules.
        """
        return Breeder(breeding_rules, self.fitting_function, self.crossover_maker)

    def create_polygons(self):
        """
        Create polygons for one solution in a simulation.
        @return: polygons for the solution.
        """
        polygons = {}
        for id, neighbours_ids in self.neighbours_ids_lists.items():
            color = choice(ALL_COLORS)
            polygon = Polygon(color, id, neighbours_ids)
            polygons[id] = polygon
        return polygons

    def create_simulation_stats(self, simulation_id):
        """
        @param simulation_id: The simulation identifier.
        @return: simulation stats for the given simulation.
        """
        return SimulationStats(simulation_id, self.fitting_function)

    def create_solutions(self, num_of_solutions):
        """
        Create solutions for a simulation.
        @param num_of_solutions: number of solutions that need to be created.
        @return: solutions.
        """
        solutions = []
        for _ in range(num_of_solutions):
            polygons = self.create_polygons()
            solution = Solution(genetic_units=polygons, fitting_function=self.fitting_function)
            solutions.append(solution)
        return solutions

    def create_simulation(self, simulation_id, num_of_solutions, mutation_maker, simulation_stats, breeder,
                          sim_viz=SimulationViz.FRONT):
        """
        Create a simulation with these specifics:
        @param simulation_id: The simulation identifier.
        @param num_of_solutions: number of solutions that need to be created.
        @param mutation_maker: making a mutation.
        @param simulation_stats: these simulation statistics.
        @param breeder: the breeder created.
        @param sim_viz: Indicates if the simulation is with a visual interface.
        @return: The created simulation.
        """
        solutions = self.create_solutions(num_of_solutions)
        simulation = Simulation(simulation_id, solutions, mutation_maker, self.fitting_function, simulation_stats,
                                breeder, sim_viz)
        return simulation

    def create_all_background_simulations(self):
        """
        Creating simulations without a visual interface.
        @return: simulations.
        """
        simulations = []
        simulations_ids = count(1)
        simulations_permutations = self.create_permutations()
        sim_viz = SimulationViz.BACK
        for sim_id, permutation in zip(simulations_ids, simulations_permutations):
            mutation_rules = self.create_mutation_rules(permutation.mutation_rate)
            mutation_maker = self.create_mutation_maker(mutation_rules)
            simulation_stats = self.create_simulation_stats(sim_id)
            breeding_rules = self.create_breading_rules(permutation.elitist_candidates, permutation.elitism_rate,
                                                        permutation.discard_rate)
            breeder = self.create_breeder(breeding_rules)
            simulation = self.create_simulation(sim_id, permutation.num_of_solutions, mutation_maker, simulation_stats,
                                                breeder, sim_viz)
            simulations.append(simulation)
            if len(simulations) == self.subprocesses:
                yield simulations 
                simulations = []
        yield simulations

    class SimulationPermutation:
        """
        creates different permutations for each simulation.
        """
        def __init__(self, num_of_solutions, elitist_candidates, elitism_rate, discard_rate, mutation_rate):
            """
            @param num_of_solutions: number of expected solutions.
            @param elitist_candidates: number of candidates to become elitists.
            @param elitism_rate: percentage of elitists in population.
            @param discard_rate: percentage of population to discard.
            @param mutation_rate: mutation rate.
            """
            self.num_of_solutions = num_of_solutions
            self.elitist_candidates = elitist_candidates
            self.elitism_rate = elitism_rate
            self.discard_rate = discard_rate
            self.mutation_rate = mutation_rate

    def create_front_simulation(self):
        """
        Create a simulation with a visual interface.
        @return: simulation created.
        """
        simulations_id = 1
        sim_viz = SimulationViz.FRONT
        mutation_rules = self.create_mutation_rules(mutation_rate=0.005)
        mutation_maker = self.create_mutation_maker(mutation_rules)
        simulation_stats = self.create_simulation_stats(simulations_id)
        breeding_rules = self.create_breading_rules(elitist_candidates=3, elitism_rate=0.05, discard_rate=0.05)
        breeder = self.create_breeder(breeding_rules)
        num_of_solutions = 150
        simulation = self.create_simulation(simulations_id, num_of_solutions, mutation_maker, simulation_stats,
                                            breeder, sim_viz)
        return simulation

    def create_permutations(self):
        """
        Create the permutations
        @return: simulation permutation.
        """
        for num_of_solutions in self.all_num_of_solutions:
            for elitist_candidates in self.all_elitist_candidates:
                for elitism_rate in self.all_elitism_rates:
                    for discard_rate in self.all_discard_rates:
                        for mutation_rate in self.all_mutation_rates:
                            for _ in range(self.num_of_revisions):  # num of revisions for each permutation
                                yield self.SimulationPermutation(num_of_solutions, elitist_candidates, elitism_rate,
                                                                 discard_rate, mutation_rate)


class Color(Enum):
    """
    Enum for colors of the polygons.
    """
    BLACK = auto()
    BLUE = auto()
    GREEN = auto()
    RED = auto()


ALL_COLORS = tuple(color for color in Color)


ENUM_TO_GUI_COLORS: Dict[Color, str] = {Color.BLACK: "black", Color.BLUE: "blue", Color.GREEN: "green", Color.RED: "red"}


class MapGUI(Frame):
    """
    GUI creator.
    """
    def __init__(self, root, simulation):
        """
        @param root: GUI window to draw on.
        @param simulation: simulation to draw.
        """
        super().__init__()
        self.simulation = simulation
        self.canvas = Canvas(root)
        self.draw_final_screen()

    def draw_final_screen(self):
        """
        The border of each polygon is given as a series of (x, y) coordinates, connected with each other neighbour
        coordinate by a line.
        """
        self.master.title("Map")   # title of the screen
        self.canvas.pack(fill=BOTH, expand=1)   # arrange objects inside the screen
        best_solution, best_score = self.find_best_solution_and_score()
        # 1
        poly_1_coords = [60, 60, 350, 60, 350, 150, 60, 150]
        enum_color = best_solution.genetic_units[1].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_1_coords, outline='black', fill=gui_color, width=2)
        # 2
        poly_2_coords = [60, 150, 350, 150, 350, 350, 260, 350, 260, 200, 60, 200]
        enum_color = best_solution.genetic_units[2].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_2_coords, outline='black', fill=gui_color, width=2)
        # 3
        poly_3_coords = [60, 200, 260, 200, 260, 440, 60, 440]
        enum_color = best_solution.genetic_units[3].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_3_coords, outline='black', fill=gui_color, width=2)
        # 4
        poly_4_coords = [60, 440, 60, 350, 110, 350, 110, 370, 220, 370, 220, 440]
        enum_color = best_solution.genetic_units[4].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_4_coords, outline='black', fill=gui_color, width=2)
        # 5
        poly_5_coords = [160, 440, 160, 400, 290, 400, 290, 440]
        enum_color = best_solution.genetic_units[5].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_5_coords, outline='black', fill=gui_color, width=2)
        # 6
        poly_6_coords = [350, 60, 640, 60, 640, 220, 350, 220]
        enum_color = best_solution.genetic_units[6].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_6_coords, outline='black', fill=gui_color, width=2)
        # 10
        poly_10_coords = [350, 220, 640, 220, 640, 450, 350, 450]
        enum_color = best_solution.genetic_units[10].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_10_coords, outline='black', fill=gui_color, width=2)
        # 8
        poly_8_coords = [350, 220, 550, 220, 550, 410, 350, 410]
        enum_color = best_solution.genetic_units[8].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_8_coords, outline='black', fill=gui_color, width=2)
        # 9
        poly_9_coords = [260, 350, 400, 350, 400, 380, 450, 380, 450, 440, 260, 440]
        enum_color = best_solution.genetic_units[9].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_9_coords, outline='black', fill=gui_color, width=2)
        # 7
        poly_7_coords = [350, 150, 500, 150, 500, 250, 350, 250]
        enum_color = best_solution.genetic_units[7].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_7_coords, outline='black', fill=gui_color, width=2)
        # 11
        poly_11_coords = [0, 0, 450, 0, 450, 60, 60, 60, 60, 440, 250, 440, 250, 500, 0, 500]
        enum_color = best_solution.genetic_units[11].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_11_coords, outline='black', fill=gui_color, width=2)
        # 12
        poly_12_coords = [450, 0, 700, 0, 700, 500, 250, 500, 250, 440, 640, 440, 640, 60, 450, 60]
        enum_color = best_solution.genetic_units[12].color
        gui_color = ENUM_TO_GUI_COLORS[enum_color]
        self.canvas.create_polygon(poly_12_coords, outline='black', fill=gui_color, width=2)
        # draw the text at the bottom of the gui.
        self.canvas.create_text(330, 520, font="Purisa", fill="black",
                                text=f"Done after {self.simulation.round} rounds     Best score:  {best_score}")
        # arrange objects on the canvas.
        self.canvas.pack(fill=BOTH, expand=1)

    def find_best_solution_and_score(self):
        """
        find the best solution created and its score - which is the best score.
        @return: best solution and best score.
        """
        best_score = MAXSIZE
        best_solution = self.simulation.solutions[0]
        for solution in self.simulation.solutions:
            score = self.simulation.fitting_function.fit_score(solution)
            if score < best_score:
                best_score = score
                best_solution = solution
        return best_solution, best_score


class Simulation:
    """
    Simulation represents the methodology of a generic genetic algorithm.

    A generic genetic algorithm is composed of the following steps:
    1) Initialization of the solutions (happens outside the simulation as the solutions are given to the simulation as
    input).
    2) Selection - a portion of solutions from the existing population is selected (based on their fitting score,
    from best to worse) to breed a new generation of solutions.
    3) Breeding the above-mentioned selected solutions, including crossover between them.
    4) Mutations, that might occur to any of the new solutions.
    5) Termination, based on the fitting function score.

    A generic simulation solves a specific problem by getting as input (a) solutions, (b) mutation_maker,
    (c) breeder and (d) fitting_function which are specific to that problem.
    In our case, the solutions represent colored maps, the breeder and mutation_maker work with discrete
    genetic units which are represented by the polygons in each solution's map, and the fitting_function evaluates a
    score of a solution based on the colors of it's polygons.
    """
    def __init__(self, simulation_id, solutions: list, mutation_maker, fitting_function, simulation_stats,
                 breeder, simulation_viz=SimulationViz.FRONT):
        """
        @param simulation_id: simulation identifier.
        @param solutions: solutions created.
        @param mutation_maker: mutation maker.
        @param fitting_function: fitting function to count fitting score.
        @param simulation_stats: simulation statistics of given simulation.
        @param breeder: breeder.
        @param simulation_viz: Indicates if the simulation is with a visual window i.e GUI.
        """
        self.simulation_id = simulation_id
        self.solutions = solutions
        self.mutation_maker = mutation_maker
        self.fitting_function = fitting_function
        self.simulation_stats = simulation_stats
        self.breeder = breeder
        self.simulation_viz = simulation_viz
        self.round = 1

    def play(self):
        """
        Play a simulation.

        Check if a legal solution exists.
        If it does, report it and end the simulation. Otherwise, play one more round.

        The "legal solution" term is a general term for a solution whose fitting score is a good enough score (as
        defined by the fitting function) in order to end the simulation.
        In our case, the fitting function literally defines a good solution as a legal solution, i.e. that its polygons
        are colored according to the rules.
        """
        legal_solution_found = False
        self.simulation_stats.add_stats(self.round, self.solutions)
        if self.simulation_viz is SimulationViz.FRONT:
            self.report_progress()
        while not legal_solution_found:
            legal_solution = self.find_solution()  # might exist already in the first round
            if legal_solution:
                legal_solution_found = True
            else:
                self.play_one_round()
        if self.simulation_viz is SimulationViz.FRONT:
            self.draw_final_screen()
        else:
            output_lines = self.report_stats()
            return output_lines

    def play_one_round(self):
        """
        Play one round of the simulation.

        Add current fitting scores to the statistics.
        Create ("breed") new solutions for the next round (there might not be another round if one of these new
        solutions is a legal solution).
        """
        new_solutions = self.breeder.breed(self.solutions)
        self.solutions.clear()
        self.solutions.extend(new_solutions)
        self.mutation_maker.mutate(self.solutions)
        self.round += 1
        self.simulation_stats.add_stats(self.round, self.solutions)
        if self.simulation_viz is SimulationViz.FRONT:
            self.report_progress()

    def draw_final_screen(self):
        """
        creating a GUI instance and drawing the final solution found on the window.
        @return: window to draw on.
        """
        root = Tk()
        MapGUI(root, self)
        root.geometry('710x540')
        root.mainloop()

    def find_solution(self):
        """
        checking if there is a legal solution in current solutions.
        @return: the legal solution if exists.
        """
        for solution in self.solutions:
            if self.fitting_function.is_legal_solution(solution):
                return solution
        return None

    def report_stats(self):
        """
        reporting simulation statistics.
        @return: lines containing the simulation statistics.
        """
        output_lines = []
        stats = self.simulation_stats.stats
        sim_id = self.simulation_id
        solutions = self.solutions
        elitist_candidates = self.breeder.breeding_rules.elitist_candidates
        elitism_rate = self.breeder.breeding_rules.elitism_rate
        discard_rate = self.breeder.breeding_rules.discard_rate
        mutation_rate = self.mutation_maker.mutation_rules.mutation_rate
        num_of_solutions = len(solutions)
        for round in range(1, self.round+1):
            scores = stats[round]  # list of scores for that round
            best_score = min(scores)
            avg_score = sum(scores) / len(solutions)
            colors = ""
            if best_score == 0:
                for solution in solutions:
                    if self.fitting_function.fit_score(solution) == 0:
                        colors = ";".join([str(solution.genetic_units[x].color.name) for x in
                                           range(1, NUM_OF_GENETIC_UNITS+1)])
            line = f"{sim_id},{round},{best_score},{avg_score},{num_of_solutions},{elitist_candidates}," \
                   f"{elitism_rate},{discard_rate},{mutation_rate},{colors}\n"
            output_lines.append(line)
        return output_lines

    def report_progress(self):
        """
        Report progress at every round to the console.
        """
        stats = self.simulation_stats.stats
        solutions = len(self.solutions)
        round = self.round
        scores = stats[round]
        best_score = min(scores)
        avg_score = sum(scores) / solutions
        line = f"Round: {round}\t\tBest Score: {best_score}\t\t Average Score: {avg_score}"
        print(line)


class SimulationStats:
    """
    This class is responsible for the simulation statistics.
    """
    def __init__(self, simulation_id, fitting_function):
        """
        @param simulation_id: simulation identifier.
        @param fitting_function: fitting function that calculates the fit score.
        """
        self.simulation_id = simulation_id
        self.fitting_function = fitting_function
        self.stats = defaultdict(list)

    def add_stats(self, round, solutions):
        """
        adding current round simulation stats to the total simulation stats.
        @param round: round number.
        @param solutions: solutions at this round.
        """
        for solution in solutions:
            solution_score = self.fitting_function.fit_score(solution)
            self.stats[round].append(solution_score)


class Breeder:
    """
    This class is responsible for the breeding.
    """
    def __init__(self, breeding_rules, fitting_function, crossover_maker):
        """
        @param breeding_rules: rules that dictate the breeding.
        @param fitting_function: function that calculates the fitting score.
        @param crossover_maker: makes the crossover.
        """
        self.breeding_rules = breeding_rules
        self.fitting_function = fitting_function
        self.crossover_maker = crossover_maker

    def breed(self, solutions):
        """
        Breed the population, according to the rules given.
        Parents are chosen randomly according to their breeding chance which was determined by score,
        a crossover will occur between them by the crossover_maker,
        elitist will be assigned as part of the new population and the rest of the population are the children produced.
        @param solutions: current solutions to breed.
        @return: new population after breeding.
        """
        # get the participants in each breading group
        elitists, crossover = self.select(solutions)
        # create a new population by elitism
        num_of_elitists_next_gen = self.breeding_rules.elitism_rate * len(solutions)
        elitists_multiplication_rate = int(num_of_elitists_next_gen / len(elitists))
        new_elitists = [[elitist for _ in range(elitists_multiplication_rate)] for elitist in elitists]
        new_elitists = list(chain.from_iterable(new_elitists))
        # create another new population by crossover
        new_crossover = []
        crossover_size_in_next_gen = len(solutions) - len(new_elitists)
        breeding_chance = self.define_breeding_chance_by_score(crossover)
        for _ in range(crossover_size_in_next_gen):
            parent_a, parent_b = np.random.choice(a=crossover, size=2, replace=False, p=breeding_chance)
            new_genetic_units = self.crossover_maker.crossover(parent_a, parent_b)
            new_solution = Solution(new_genetic_units, self.fitting_function)
            new_crossover.append(new_solution)
        # merge those new populations into list, and return that list
        new_population = new_elitists
        new_population.extend(new_crossover)
        return new_population

    def select(self, solutions):
        """
        Select the possible participants for each breading group, based on their fitting score.
        @param solutions: solutions to select from.
        @return: the elitists and crossover groups to breed.
        """
        solutions = self.sort_solutions(solutions)
        # define coordinates for the two groups
        elitists_coords = [x for x in range(self.breeding_rules.elitist_candidates)]
        first_discarded_solution = int(len(solutions) - (len(solutions) * self.breeding_rules.discard_rate))
        crossover_coords = [x for x in range(first_discarded_solution)]
        # fill each breeding group with its possible participants, based on the coordinates defined above
        elitists = [solutions[x] for x in elitists_coords]
        crossover = [solutions[x] for x in crossover_coords]
        return elitists, crossover

    def define_breeding_chance_by_score(self, crossover):
        """
        Define probability for each solution to breed by crossover based on its score.
        @param crossover: group that will participate in crossover.
        @return: breeding chance for each solution.
        """
        breeding_chance = []
        zones_num = self.breeding_rules.chance_zones
        zones = range(zones_num)
        zone_size = int(len(crossover) / zones_num)
        first_chance = int(self.breeding_rules.first_zone_chance * 100)
        last_chance = int(self.breeding_rules.last_zone_chance * 100)
        chance_step = int((last_chance - first_chance) / zones_num)
        chances = range(first_chance, last_chance, chance_step)
        for zone, chance in zip(zones, chances):
            chance = chance / 100
            for _ in range(zone_size):
                breeding_chance.append(chance)
        len_diff = len(breeding_chance) - len(crossover)
        if 0 < len_diff:
            breeding_chance = breeding_chance[:len(breeding_chance) - len_diff]
        if 0 > len_diff:
            len_diff = -len_diff
            breeding_chance.extend(breeding_chance[-1] for _ in range(len_diff))
        chance_sum = sum(breeding_chance)
        breeding_chance = [chance/chance_sum for chance in breeding_chance]
        return breeding_chance

    def sort_solutions(self, solutions):
        """
        Sort solutions from best to worse, according to ScoresSortingOrder.
        In our case we sorted in accending order, meaning the best score is the lowest.
        @param solutions: solutions to sort.
        @return: sorted solutions from best to worse.
        """
        if self.breeding_rules.sorting_order is ScoresSortingOrder.ASCENDING:
            reverse = False
        else:
            reverse = True
        return sorted(solutions, reverse=reverse, key=lambda solution: solution.score)


class ScoresSortingOrder(Enum):
    """
    Enum for acceding and descending order.
    """
    ASCENDING = auto()
    DESCENDING = auto()


class BreedingRules:
    """
    Defining the rules for breeding.
    """
    def __init__(self, sorting_order, elitist_candidates, elitism_rate, discard_rate, chance_zones,
                 first_breeding_zone_chance, last_breeding_zone_chance):
        """
        @param sorting_order: which order the scores are sorted. in our case- ascending.
        @param elitist_candidates: number of candidates for elitism.
        @param elitism_rate: percentage of elitism in the population.
        @param discard_rate: percentage of discarded members from the population (with the lowest scores).
        @param chance_zones: number of zones,
        so that in each zone the breeding chance is the same for every member in the group.
        @param first_breeding_zone_chance: first part of the population (i.e first indexes) are the ones with the best scores.
        Therefore they will have the highest breeding chance.
        @param last_breeding_zone_chance: last part of the population (i.e last indexes) are the ones with the worst scores.
        Therefore they will have the lowest breeding chance.
        """
        self.sorting_order = sorting_order
        self.elitist_candidates = elitist_candidates
        self.elitism_rate = elitism_rate
        self.discard_rate = discard_rate
        self.chance_zones = chance_zones  # num of breeding chance zones
        self.first_zone_chance = first_breeding_zone_chance
        self.last_zone_chance = last_breeding_zone_chance


class Solution:
    """
    The genetic units each represent a solution, and they each have a score, according to the fitting_function.
    """
    def __init__(self, genetic_units: dict, fitting_function):
        self.genetic_units = genetic_units  # in our case the genetic_units are Polygons
        self.score = fitting_function.fit_score(self)


class FittingFunction:
    """
    This class is responsible for assessing the fitting score for each genetic unit.
    """
    def __init__(self, sorting_order):
        """
        @param sorting_order: sorting order of the solutions.
        """
        self.sorting_order = sorting_order

    def fit_score(self, solution):
        """
        A score is calculated according to the sorting order, so that the best will be the first in both cases.
        Presence of a illegal neighbour is an occurrence of two neighbours with the same color that share an edge.
        Presence of a legal neighbour is an occurrence of two neighbours with different colors that share an edge.
        @param solution:
        @return: score of the solution.
        """
        illegal_neighbours = 0
        legal_neighbours = 0
        for polygon in solution.genetic_units.values():
            for neighbour_id in polygon.neighbours_ids:
                if polygon.color is solution.genetic_units[neighbour_id].color:
                    illegal_neighbours += 1
                else:
                    legal_neighbours += 1
        if self.sorting_order is ScoresSortingOrder.ASCENDING:
            return illegal_neighbours
        else:
            return legal_neighbours

    def is_legal_solution(self, solution):
        """
        determine if the solution received is legal.
        @param solution: solution to asses.
        @return: fit score.
        """
        if self.sorting_order is ScoresSortingOrder.ASCENDING:
            return self.fit_score(solution) == 0
        else:
            return self.fit_score(solution) == sum(x for x in range(1, 12))


class Polygon:
    """
    This class is responsible for the polygons. all polygons in one solution are a genetic unit.
    neighbours are polygons who can brake the rules, i.e polygons that share an edge with another polygon.
    they are defined according to the given map, but only those found downstream in
    topological sort are given for each polygon, so 1 isn't in the neighbours list of 2
    (that way we check each couple of neighbours only once)
    """
    def __init__(self, color, id, neighbours_ids: list):
        """
        @param color: color of the polygon.
        @param id: id of the polygon.
        @param neighbours_ids: neighbour ids of the polygon.
        """
        self.color = color
        self.id = id  # range(1, NUM_OF_GENETIC_UNITS+1)
        self.neighbours_ids = neighbours_ids

    def __repr__(self):
        """
        @return: polygon with its id ang color.
        """
        return f"< Polygon  {self.id}   {self.color} >"



class CrossoverMaker:
    """
    Making he crossover for two parents.
    """
    def crossover(self, parent_1, parent_2):
        """
        Create the polygons of a new child by crossover between his parents parent_1 and parent_2.

        Randomly select one parent as parent_a, and the other as parent_b.
        Take all polygons from parent_a, and replaces polygons in range(start, end) with those of parent_b, thus
        creating a crossover.
        Return the polygons (a dict) of the new child.
        @param parent_1: parent that was selected according to its breeding chance.
        @param parent_2: parent that was selected according to its breeding chance..
        @return:
        """
        start = randrange(1, NUM_OF_GENETIC_UNITS + 1)
        end = randrange(1, NUM_OF_GENETIC_UNITS + 1)
        if end < start:
            start, end = end, start
        if start == end:
            end = NUM_OF_GENETIC_UNITS + 1
        if randrange(1, 3) == 1:
            parent_a = parent_1
            parent_b = parent_2
        else:
            parent_a = parent_2
            parent_b = parent_1
        crossover_of_genetic_units = {genetic_unit_num: parent_a.genetic_units[genetic_unit_num] for genetic_unit_num
                                      in parent_a.genetic_units}
        for genetic_unit_num in range(start, end):
            crossover_of_genetic_units[genetic_unit_num] = parent_b.genetic_units[genetic_unit_num]
        return crossover_of_genetic_units


class MutationMaker:
    """
    Create a mutation, according to the mutation rules, on the current solutions.
    """
    def __init__(self, mutation_rules):
        """
        @param mutation_rules: rules for mutation.
        """
        self.mutation_rules = mutation_rules

    def mutate(self, solutions):
        """
        Mutate some of the solutions.

        Define k, which is the number of solutions that will undergo a mutation process.
        Sample k solutions, and randomly mutate one polygon's color of each.
        @param solutions: solutions to mutate.
        """
        k = ceil(self.mutation_rules.mutation_rate * len(solutions))
        solutions_to_mutate = sample(solutions, k)
        for solution in solutions_to_mutate:
            new_color = choice(ALL_COLORS)
            polygon_num = randrange(1, NUM_OF_GENETIC_UNITS + 1)
            polygon = solution.genetic_units[polygon_num]
            polygon.color = new_color


class MutationRules:
    """
    Mutation rules are according to mutation rate.
    """
    def __init__(self, mutation_rate):
        """
        @param mutation_rate: percentage to mutate.
        """
        self.mutation_rate = mutation_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--out_dir")
    parser.add_argument("-s", "--subprocesses", default=1, type=int)
    parser.add_argument("-v", "--simulation_viz", default="front", choices=("front", "back"))
    args = parser.parse_args()
    out_dir = args.out_dir
    subprocesses = args.subprocesses
    if args.simulation_viz == "front":
        simulation_viz = SimulationViz.FRONT
    else:
        simulation_viz = SimulationViz.BACK
    Game.play(out_dir, subprocesses, simulation_viz)
