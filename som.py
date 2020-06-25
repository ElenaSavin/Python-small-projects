from multiprocessing import Pool
import argparse
from random import randint, shuffle
import numpy as np
import math
from sys import maxsize as MAXSIZE
from enum import Enum, auto
from itertools import count
from datetime import datetime
import os
from tkinter import *
from collections import defaultdict, Counter, namedtuple


class Game:
    @staticmethod
    def play(out_dir, subprocesses, simulation_viz, digits_file, required_rounds_without_change, threshold_change_rate,
             acceptable_rounds_without_change, acceptable_round):
        """
        Play all kinds of different simulations.
        @param out_dir: directory of the output file with simulation stats.
        @param subprocesses: number of subprocesses.
        @param simulation_viz: indicates if the simulation is with a visual window i.e GUI.
        @param digits_file: imported file of input Digits.
        @param required_rounds_without_change: number of round should the simulation play without changes.
        @param threshold_change_rate:
        @param acceptable_rounds_without_change: number of rounds without change that are acceptable to stop the simulation.
        @param acceptable_round: round number when the simulation stopped after acceptable rounds without change.
        """
        simulator = Simulator(digits_file, required_rounds_without_change, threshold_change_rate,
                              acceptable_rounds_without_change, acceptable_round)
        if simulation_viz is SimulationViz.FRONT:
            simulator.simulate_front_simulation()
        else:
            simulator.simulate_background_simulations(out_dir, subprocesses)


class Simulator:
    """Play simulation(s), and write statistics and final solutions of background simulations to files."""

    def __init__(self, digits_file, required_rounds_without_change, threshold_change_rate,
                 acceptable_rounds_without_change, acceptable_round):
        """
        @param digits_file: imported file of input Digits.
        @param required_rounds_without_change: number of round should the simulation play without changes.
        @param threshold_change_rate: minimal percentage of digits positions changed in order to consider the round as a round with change.
        @param acceptable_rounds_without_change: number of rounds without change that are acceptable to stop the simulation.
        @param acceptable_round: round number when the simulation stopped after acceptable rounds without change.        """
        self.digits_file = digits_file
        self.required_rounds_without_change = required_rounds_without_change
        self.threshold_change_rate = threshold_change_rate
        self.acceptable_rounds_without_change = acceptable_rounds_without_change
        self.acceptable_round = acceptable_round

    def simulate_front_simulation(self):
        """
        Play a GUI simulation.
        """
        factory = SimulationFactory(self.digits_file, self.required_rounds_without_change, self.threshold_change_rate,
                                    self.acceptable_rounds_without_change, self.acceptable_round)
        simulation = factory.create_front_simulation()
        simulation.play()

    def simulate_background_simulations(self, out_dir, subprocesses):
        """
        Run simulations in parallel and write their statistics and final solution to files.

        Get all_simulations_by_groups, a list of lists of simulations, where each inner list's size is the number of
        subprocesses to run in parallel.
        Then run each group of simulations (each inner list in all_simulations_by_groups) in parallel.
        @param out_dir: directory of output files.
        @param subprocesses: number of sub processes.
        """
        all_output_lines = []
        all_digits = []
        all_sim_ids = []
        factory = SimulationFactory(self.digits_file, self.required_rounds_without_change, self.threshold_change_rate,
                                    self.acceptable_rounds_without_change, self.acceptable_round, subprocesses)
        all_simulations_by_groups = factory.create_background_simulations()
        for group_num, group_of_simulations in enumerate(all_simulations_by_groups):
            self.report_progress(group_num, len(group_of_simulations))
            with Pool() as pool:
                simulations_outputs = pool.map(Simulator.start_background_simulation, group_of_simulations)
                all_output_lines.extend([simulation_output.output_lines for simulation_output in simulations_outputs])
                all_digits.extend([simulation_output.digits for simulation_output in simulations_outputs])
                all_sim_ids.extend([simulation_output.simulation_id for simulation_output in simulations_outputs])
        self.report_end()
        self.write_background_stats(out_dir, all_output_lines)
        self.write_solved_maps(out_dir, all_digits, all_sim_ids)

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
            col_names = "sim_id,digits_order,round,quantization_error,topological_error,rounds_without_change\n"
            file.write(col_names)
            for lines_of_one_simulation in all_output_lines:
                for line in lines_of_one_simulation:
                    file.write(line)

    def write_solved_maps(self, out_dir, all_digits, all_sim_ids):
        """
        Write solved maps in output file.
        @param out_dir: directory of the output file with solved maps.
        @param all_digits: all digits to write.
        @param all_sim_ids: all simulations ids.
        """
        out_file = os.path.join(out_dir, "solved_maps.csv")
        with open(out_file, "w") as file:
            locations = [f"{i}_{j}" for i in range(6) for j in range(6)]
            col_names = f"sim_id,{','.join(locations)}\n"
            file.write(col_names)
            for digits, sim_id in zip(all_digits, all_sim_ids):
                digits_by_neuron = defaultdict(str)
                for digit in digits:
                    row, col = digit.row, digit.col
                    location = f"{row}_{col}"
                    if len(digits_by_neuron[location]) == 1:
                        digits_by_neuron[location] = digit.value
                    else:
                        digits_by_neuron[location] += f";{digit.value}"
                line = f"{sim_id},{','.join(digits_by_neuron[location] for location in locations)}\n"
                file.write(line)

    @classmethod
    def start_background_simulation(cls, simulation):
        """
        Used in order to achieve multiprocessing with pool.map.
        @param simulation: The simulation running.
        @return: simulation_output
        """
        simulation_output = simulation.play()
        return simulation_output


class SimulationViz(Enum):
    """Denotes the visualization of a simulation.

    For a number of simulations running in parallel, the first one is FRONT and is printed (drawn) to the screen, and
    the other ones are BACK (because they are run and calculated without graphical representation)."""
    FRONT = auto()
    BACK = auto()


class SimulationFactory:
    """
        This is a simulation factory class. creates simulations.
    """
    def __init__(self, digits_file, required_rounds_without_change, threshold_change_rate,
                 acceptable_rounds_without_change, acceptable_round, subprocesses=1, digits_order="C"):
        """
        @param digits_file: imported file of input Digits.
        @param required_rounds_without_change: number of round should the simulation play without changes.
        @param threshold_change_rate: minimal percentage of digits positions changed in order to consider the round as a round with change.
        @param acceptable_rounds_without_change: number of rounds without change that are acceptable to stop the simulation.
        @param acceptable_round: round number when the simulation stopped after acceptable rounds without change.
        @param subprocesses: number of sub processes.
        @param digits_order: order of the digits.
        """
        self.digits_file = digits_file
        self.subprocesses = subprocesses
        self.digits_order = digits_order
        self.num_of_revisions = 50
        self.required_rounds_without_change = required_rounds_without_change
        self.threshold_change_rate = threshold_change_rate
        self.acceptable_rounds_without_change = acceptable_rounds_without_change
        self.acceptable_round = acceptable_round

    def create_digits(self):
        """
        Creating a digits matrix of the input digits.
        @return: Digits.
        """
        with open(self.digits_file) as file:
            digits = []
            matrix_lines = {}
            line_num = 0
            value = 0
            digits_with_same_value = 0
            for line in file:
                line = line.rstrip("\n ")
                if line_num == 10:
                    matrix = np.array([matrix_lines[x] for x in range(10)])
                    digit = Digit(value, matrix)
                    digits.append(digit)
                    matrix_lines = {}
                    line_num = 0
                    digits_with_same_value += 1
                    if digits_with_same_value == 10:
                        value += 1
                        digits_with_same_value = 0
                if 0 < len(line):
                    bits = []
                    for bit in line:
                        if bit == "0":
                            bits.append(0)
                        else:
                            bits.append(1)
                    matrix_lines[line_num] = np.array(bits)
                    line_num += 1
        if len(digits) != 100:
            raise Exception("Input file read wrong")  # sanity check
        return digits

    def create_background_simulations(self):
        """
        Creating simulations without a visual interface.
        @return: simulations.
        """
        simulations = []
        simulation_id = 1
        simulation_viz = SimulationViz.BACK
        digits = self.create_digits()
        for digits_order in ["C", "F"]:
            for _ in range(self.num_of_revisions):
                map = Map(digits, digits_order)
                simulation_stats = SimulationStats()
                simulation = Simulation(simulation_id, map, simulation_stats, self.required_rounds_without_change,
                                        self.threshold_change_rate, self.acceptable_rounds_without_change,
                                        self.acceptable_round, simulation_viz)
                simulation_id += 1
                simulations.append(simulation)
                if len(simulations) == self.subprocesses:
                    yield simulations
                    simulations = []
        yield simulations

    def create_front_simulation(self):
        """
        Create a simulation with a visual interface.
        @return: simulation created.
        """
        simulation_id = 1
        digits = self.create_digits()
        map = Map(digits)
        simulation_stats = SimulationStats()
        simulation_viz = SimulationViz.FRONT
        simulation = Simulation(simulation_id, map, simulation_stats, self.required_rounds_without_change,
                                self.threshold_change_rate, self.acceptable_rounds_without_change,
                                self.acceptable_round, simulation_viz)
        return simulation


class Color(Enum):
    """
    Enum for colors of the polygons.
    """
    BLACK = auto()
    BLUE = auto()
    GREEN = auto()
    RED = auto()


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
        self.master = root
        self.canvas = Canvas(root)
        self.draw_final_screen()

    def draw_final_screen(self):
        """
        Draw the final mapped digits.
        """
        w = Canvas(self.master, width=980, height=805)
        self.master.title("Neural Network Digits")  # Add a title
        colors = {1: "indianred", 2: "coral", 3: "palegoldenrod", 4: "lightgreen", 5: "darkturquoise",
                  6: "darkcyan", 7: "royalblue", 8: "darkslateblue", 9: "darkviolet", 0: "gray", -1: "white"}

        digits_in_neurons = [(digit.value, (digit.row, digit.col)) for digit in self.simulation.map.digits]
        values = [i[0] for i in digits_in_neurons]  # digits
        locations = [i[1] for i in digits_in_neurons]  # indexes
        unique_locations = list(set(locations))  # unique mapped locations.

        for i in range(6):
            for j in range(6):
                if (i, j) in unique_locations:
                    # number of locations for each digit.
                    values_in_location = Counter([value for value, location in zip(values, locations)
                                                  if location == (i, j)])
                    digits = [digit for digit in values_in_location.keys()]  # list of unique digits mapped.
                    total_mapped = sum(values_in_location.values())  # how many digits in total were mapped.
                    # digits_portion == portion of mapped digits in specific neuron
                    digits_portion = [round(values_in_location[digit] / total_mapped * 100) for digit in digits]
                    digits_in_rectangles = []
                    # fill an digits_in_neurons in
                    for n in range(len(digits)):
                        digits_in_rectangles.extend([digits[n]] * digits_portion[n])
                    shuffle(digits_in_rectangles)
                    digits_in_rectangles = np.reshape(digits_in_rectangles, newshape=(10, 10))
                else:
                    digits_in_rectangles = np.full((10, 10), -1, dtype=int)
                for k in range(10):
                    for l in range(10):
                        draw_digit = digits_in_rectangles[k, l]
                        color = colors[draw_digit]
                        w.create_rectangle(13.4*k + 134*i,
                                           13.4*l + 134*j,
                                           13.4*k + 134*i + 13.4,
                                           13.4*l + 134*j + 13.4,
                                           fill=color, outline=color)

        text_quant = f"Quantization \nerror:  {self.simulation.map.quantization_error()}"
        text_topo = f"Topological \nerror:  {self.simulation.map.topological_error()}"
        w.create_text(900, 500, fill="black", font="Ariel 18 italic", text=text_topo)
        w.create_text(900, 580, fill="black", font="Ariel 18 italic", text=text_quant)
        w.create_text(850, 300, fill="black", font="Ariel 20 italic", text="0 \n\n1 \n\n2 \n\n3 \n\n4")
        w.create_text(930, 300, fill="black", font="Ariel 20 italic", text="5 \n\n6 \n\n7 \n\n8 \n\n9 ")

        for num in range(5):
            color = colors[num]
            w.create_rectangle(865, 165 + 62 * num, 885, 185 + 62 * num, fill=color, outline="black")
        for num in range(5, 10):
            color = colors[num]
            w.create_rectangle(940, 165 + 62 * (num - 5), 960, 185 + 62 * (num - 5), fill=color, outline="black")

        w.pack()


SimulationOutput = namedtuple("SimulationOutput", ["output_lines", "digits", "simulation_id"])


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
    def __init__(self, simulation_id, map, simulation_stats, required_rounds_without_change, threshold_change_rate,
                 acceptable_rounds_without_change, acceptable_round, simulation_viz=SimulationViz.FRONT):
        """
        @param simulation_id: simulation identifier.
        @param map: SOM network.
        @param simulation_stats: simulation statistics of given simulation.
        @param required_rounds_without_change: principle requirement for rounds without change
        (as long as round < acceptable round)
        @param threshold_change_rate: minimal percentage of digits positions changed
        in order to consider the round as a round with change.
        @param acceptable_rounds_without_change: number of rounds without change
        that are acceptable to stop the simulation.
        @param acceptable_round: round number when the simulation stopped after acceptable rounds without change.
        @param simulation_viz: Indicates if the simulation is with a visual window i.e GUI.
        """
        self.simulation_id = simulation_id
        self.map = map
        self.simulation_stats = simulation_stats
        self.required_rounds_without_change = required_rounds_without_change
        self.threshold_change_rate = threshold_change_rate
        self.round = 1
        self.rounds_without_change = 0
        self.acceptable_rounds_without_change = acceptable_rounds_without_change
        self.acceptable_round = acceptable_round
        self.simulation_viz = simulation_viz

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
        self.simulation_stats.add_stats(self.round, self.map.quantization_error(), self.map.topological_error(),
                                        self.rounds_without_change)
        if self.simulation_viz is SimulationViz.FRONT:
            self.report_progress()
        while not legal_solution_found:
            legal_solution = self.legal_solution()
            if legal_solution:
                legal_solution_found = True
            else:
                self.play_one_round()
        if self.simulation_viz is SimulationViz.FRONT:
            pass
            self.draw_final_screen()
        else:
            output = SimulationOutput(self.report_stats(), self.map.digits, self.simulation_id)
            return output

    def play_one_round(self):
        """
        Play one round of the simulation.
        """
        positions_changed = self.map.change_positions()
        print(f"positions_changed: {positions_changed}")
        if positions_changed <= 100 * self.threshold_change_rate:
            self.rounds_without_change += 1
        else:
            self.rounds_without_change = 0
        self.round += 1
        self.simulation_stats.add_stats(self.round, self.map.quantization_error(), self.map.topological_error(),
                                        self.rounds_without_change)
        if self.simulation_viz is SimulationViz.FRONT:
            self.report_progress()

    def draw_final_screen(self):
        """
        creating a GUI instance and drawing the final solution found on the window.
        """
        root = Tk()
        MapGUI(root, self)
        root.mainloop()

    def legal_solution(self):
        """
        Checking if there is a legal solution in current solutions.
        @return: True if a legal solution exists, False otherwise.
        """
        return ((self.rounds_without_change == self.required_rounds_without_change) or
                (self.rounds_without_change == self.acceptable_rounds_without_change and
                 self.round >= self.acceptable_round))

    def report_stats(self):
        """
        reporting simulation statistics.
        @return: lines containing the simulation statistics.
        """
        output_lines = []
        quantization_stats = self.simulation_stats.quantization_stats
        topological_stats = self.simulation_stats.topological_stats
        all_rounds_without_change = self.simulation_stats.rounds_without_change
        sim_id = self.simulation_id
        digits_order = self.map.digits_order
        for round in range(1, self.round+1):
            quantization_error = quantization_stats[round]
            topological_error = topological_stats[round]
            rounds_without_change = all_rounds_without_change[round]
            line = f"{sim_id},{digits_order},{round},{quantization_error},{topological_error},{rounds_without_change}\n"
            output_lines.append(line)
        return output_lines

    def report_progress(self):
        """
        Report progress at every round to the console.
        """
        round = self.round
        quantization_error = self.simulation_stats.quantization_stats[round]
        topological_error = self.simulation_stats.topological_stats[round]
        rounds_without_change = self.simulation_stats.rounds_without_change[round]
        line = f"Round: {round}\t\tQuantization Error: {quantization_error}\t\t" \
               f"Topological Error: {topological_error}\t\tRounds Without Change: {rounds_without_change}"
        print(line)


class SimulationStats:
    """SimulationStats is responsible for the simulation statistics."""
    def __init__(self):
        self.quantization_stats = {}
        self.topological_stats = {}
        self.rounds_without_change = {}

    def add_stats(self, round, quantization_error, topological_error, rounds_without_change):
        """
        Adding current round simulation stats to the total simulation stats.
        @param round: round number.
        @param quantization_error: quantization error calculated.
        @param topological_error: topological error calculated.
        @param rounds_without_change: number of rounds without change.
        """
        self.quantization_stats[round] = quantization_error
        self.topological_stats[round] = topological_error
        self.rounds_without_change[round] = rounds_without_change


class Matrix:
    """
    This class is responsible for the matrices.
    """
    def __init__(self, dim):
        """
        @param dim: dimension of the matrix.
        """
        self.grid = [[] for _ in range(dim)]  # outer list - rows, inner lists - cols

    def get(self, row, col):
        """
        Get content of cell in given row, col.

        @param row: row of the cell
        @param col: col of the cell
        @return: the value in that cell
        """
        return self.grid[row][col]

    def append(self, row, value):
        """
        Add value to inner list of row.

        Used when creating initializing the matrix inner lists, e.g., by randomized ndarrays that represent neurons
        of lists of mapped digits to neurons.

        @param row: row of inner list.
        @param value: value to append to inner list of row.
        """
        self.grid[row].append(value)

    def assign(self, row, col, value):
        """
        Assign value to a cell in given row, col.

        @param row: row of the cell
        @param col: col of the cell
        @param value: value to be assigned
        """
        self.grid[row][col] = value


class Digit:
    """
    This class is responsible for the digits.
    """
    def __init__(self, value, matrix: np.ndarray):
        """
        @param value: range(0, 10), only used for graphical representation.
        @param matrix: matrix representing the digit.
        """
        self.value = value
        self.matrix = matrix
        self.row = 0
        self.col = 0


class Map:
    """Map represents a SOM network."""
    def __init__(self, digits, digits_order="C"):
        """
        @param digits: digits in the network.
        @param digits_order: order of the digits.
        """
        self.digits = digits
        self.digits_order = digits_order
        self.neuronal_matrices = self.initialize_neuronal_matrices()
        self.learning_constant = 0.5
        self.neighbourhood_relations = [0.4, 0.2, 0.03]

    def initialize_neuronal_matrices(self):
        """
        Initializing the neuronal matrices.
        @return: neuronal matrices.
        """
        neuronal_matrices = Matrix(6)
        for row in range(6):
            for _ in range(6):
                inner_mat = np.array([np.array([randint(0, 1) for _ in range(10)]) for _ in range(10)])
                neuronal_matrices.append(row, inner_mat)
        return neuronal_matrices

    def change_positions(self):
        """
        Change positions of digits in neurons.
        Update neuronal matrices.
        @return: number of digits that changed positions.
        """
        positions_changed = 0
        shuffle(self.digits)
        for digit in self.digits:
            best_neuron_coords, _ = self.two_closest_neurons(digit)
            new_row, new_col = best_neuron_coords
            if new_row != digit.row or new_col != digit.col:
                digit.row, digit.col = new_row, new_col
                positions_changed += 1
            self.update_neuronal_matrices(digit)
        return positions_changed

    def update_neuronal_matrices(self, digit):
        """
        Update matrices of all neurons near digit.
        @param digit: given digit.
        """
        digit_matrix = digit.matrix
        chosen_neuron = digit.row, digit.col
        close_neighbours, distant_neighbours = self.find_neuron_neighbours(digit.row, digit.col)
        neurons_and_relations = [(chosen_neuron, 0)]
        neurons_and_relations.extend([(neuron, 1) for neuron in close_neighbours])
        neurons_and_relations.extend([(neuron, 2) for neuron in distant_neighbours])
        for neuron_coords, relation in neurons_and_relations:
            neuron_row, neuron_col = neuron_coords
            neuronal_matrix = self.neuronal_matrices.get(neuron_row, neuron_col)
            new_neuronal_matrix = self.update_function(digit_matrix, neuronal_matrix, relation)
            self.neuronal_matrices.assign(neuron_row, neuron_col, new_neuronal_matrix)

    def update_function(self, digit_matrix: np.ndarray, neuronal_matrix: np.ndarray, neighbourhood):
        """
        Update matrix of a neuron according to the matrix of digit, and their proximity.
        @param digit_matrix: matrix of digits.
        @param neuronal_matrix: matrix where each cell is a neuron.
        @param neighbourhood: neuron neighbourhood.
        @return: updated neuron matrix.
        """
        digit_vector = digit_matrix.flatten()
        neuron_vector = neuronal_matrix.flatten()
        new_neuron_vector = np.zeros(shape=(1, 100))
        for i, d_bit, n_bit in zip(count(0), digit_vector, neuron_vector):
            new_bit = n_bit + self.learning_constant * self.neighbourhood_relations[neighbourhood] * (d_bit - n_bit)
            new_neuron_vector[0, i] = new_bit
        new_neuron_matrix = np.reshape(new_neuron_vector, newshape=(10, 10))
        return new_neuron_matrix

    def quantization_error(self):
        """
        Calculate the average distance between a digit and the neuron representing it.
        """
        total_distance = 0
        for digit in self.digits:
            row, col = digit.row, digit.col
            distance = self.distance_to_neuron(digit, row, col)
            total_distance += distance
        quantization_error = round((total_distance / 100), 4)
        return quantization_error

    def topological_error(self):
        """
        Calculate percentage of bad mappings, i.e., digits whose second-best neuron is not near them.
        """
        bad_mappings = 0
        for digit in self.digits:
            row, col = digit.row, digit.col
            _, second_best_neuron_coords = self.two_closest_neurons(digit)
            close_neighbours, _ = self.find_neuron_neighbours(row, col)
            if second_best_neuron_coords != (row, col) and second_best_neuron_coords not in close_neighbours:
                bad_mappings += 1
        topological_error = round(((bad_mappings / 36) * 100), 4)
        return topological_error

    def find_neuron_neighbours(self, row, col):
        """
        find neuron close- first degree and distant-second degree neighbours by indices.
        @param row: neurons row number.
        @param col: neurons column number.
        @return: close and distant neighbours of given neuron.
        """
        possible_close_neighbours = [(row-1, col-1), (row, col-1), (row+1, col-1), (row-1, col),
                                     (row+1, col), (row-1, col+1), (row, col+1), (row+1, col+1)]
        possible_distant_neighbours = [(row-2, col-2), (row-2, col-1), (row-2, col), (row-2, col+1), (row-2, col+2),
                                       (row+2, col-2), (row+2, col-1), (row+2, col), (row+2, col+1), (row+2, col+2),
                                       (row-1, col-2), (row, col-2), (row+1, col-2),
                                       (row-1, col+2), (row, col+2), (row+1, col+2)]
        close_neighbours = [(row, col) for row, col in possible_close_neighbours if self.index_in_range(row, col)]
        distant_neighbours = [(row, col) for row, col in possible_distant_neighbours if self.index_in_range(row, col)]
        return close_neighbours, distant_neighbours

    def index_in_range(self, row, col):
        """
        Check if (row, col) is a legal index in the networks map.
        @param row: row to check.
        @param col: column to check
        @return: True if the indices are compatible, False otherwise.
        """
        return 0 <= row <= 5 and 0 <= col <= 5

    def two_closest_neurons(self, digit):
        """
        Find two closest neurons to digit.

        @param digit: input digit
        @type digit: Digit
        @return: rows and cols of two closest neurons to digit
        @rtype: tuple
        """
        row_1, col_1 = 0, 0
        shortest_distance = MAXSIZE
        row_2, col_2 = 0, 0
        second_shortest_distance = MAXSIZE
        first_time = True
        for i in range(6):
            for j in range(6):
                distance = self.distance_to_neuron(digit, i, j)
                if second_shortest_distance < distance:
                    continue
                if shortest_distance < distance < second_shortest_distance:
                    row_2, col_2 = i, j
                    second_shortest_distance = distance
                # distance <= shortest_distance
                else:
                    # distance < shortest_distance == MAXSIZE
                    if first_time:
                        row_1, col_1 = i, j
                        shortest_distance = distance
                        first_time = False
                    # distance <= shortest_distance <= second_shortest_distance < MAXSIZE
                    else:
                        if distance < shortest_distance:
                            row_2, col_2 = row_1, col_1
                            row_1, col_1 = i, j
                            second_shortest_distance = shortest_distance
                            shortest_distance = distance
                        else:
                            # shortest_distance < distance < second_shortest_distance
                            if distance < second_shortest_distance:
                                row_2, col_2 = i, j
                                second_shortest_distance = distance
        return (row_1, col_1), (row_2, col_2)  # return rows and cols of two closest neurons

    def distance_to_neuron(self, digit, row, col):
        """
        Calculate the euclidean distance between a digit matrix and neuron in (row, col) matrix.

        @param digit: input digit
        @param row: row of neuron.
        @param col: column of neuron.
        @return: euclidean distance between digit and neuron
        @rtype: float
        """
        digit_vector = digit.matrix.flatten(order=self.digits_order)
        neuron_vector = self.neuronal_matrices.get(row, col).flatten(order=self.digits_order)
        distance = math.sqrt(sum((d_bit - n_bit)**2 for d_bit, n_bit in zip(digit_vector, neuron_vector)))
        return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--out_dir")
    parser.add_argument("-s", "--subprocesses", default=1, type=int)
    parser.add_argument("-v", "--simulation_viz", default="front", choices=("front", "back"))
    parser.add_argument("-f", "--digits_file", default="Digits_Ex3.txt")
    parser.add_argument("-r", "--required_rounds_without_change", default=7, type=int)
    parser.add_argument("-t", "--threshold_change_rate", default=0.05, type=float)
    parser.add_argument("--acceptable_rounds_without_change", default=5)
    parser.add_argument("--acceptable_round", default=1000)
    args = parser.parse_args()
    out_dir = args.out_dir
    subprocesses = args.subprocesses
    digits_file = args.digits_file
    required_rounds_without_change = args.required_rounds_without_change
    threshold_change_rate = args.threshold_change_rate
    acceptable_rounds_without_change = args.acceptable_rounds_without_change
    acceptable_round = args.acceptable_round
    if args.simulation_viz == "front":
        simulation_viz = SimulationViz.FRONT
    else:
        simulation_viz = SimulationViz.BACK
    Game.play(out_dir, subprocesses, simulation_viz, digits_file, required_rounds_without_change, threshold_change_rate,
              acceptable_rounds_without_change, acceptable_round)
