import random as rand
import numpy as np
import pyglet as pyglet
from tkinter import *
import tkinter.messagebox
import random

WINDOW_SIZE = (600, 610)
CELL_SIZE = int(WINDOW_SIZE[0]/200)


class StartWindow:
    """
    This class creates an opening window to receive user input for:
    number of creatures, probability to infect, isolation factor K.
    """
    def __init__(self):
        self.tk = Tk()
        self.tk.title("Cellular Automata input")
        self.N = IntVar()
        self.P = DoubleVar()
        self.K = IntVar()
        self.label_1 = Label(self.tk, text="number of creatures: ")
        self.label_2 = Label(self.tk, text="probability to infect: ")
        self.label_3 = Label(self.tk, text="isolation factor K: ")
        self.entry_1 = Entry(self.tk, textvariable=self.N)
        self.entry_2 = Entry(self.tk, textvariable=self.P)
        self.entry_3 = Entry(self.tk, textvariable=self.K)
        self.num = 0
        self.prob = 0
        self.iso = 0

    def widgets(self):
        """
        creating the text, entry place and button and placing them in the window.
        """
        self.label_1.grid(row=0)
        self.label_2.grid(row=1)
        self.label_3.grid(row=2)
        self.entry_1.grid(row=0, column=1)
        self.entry_2.grid(row=1, column=1)
        self.entry_3.grid(row=2, column=1)
        button = Button(self.tk, text='Done', activebackground="red", activeforeground="blue", command=self.get_vals)
        button.grid(row=4, column=1)

    def get_vals(self):
        """
        checking validity of the input and if valid- storing in variables and closing window.
        otherwise a warning popup window will appear.
        """
        while True:
            try:
                self.num = int(self.entry_1.get())
                self.prob = float(self.entry_2.get())
                self.iso = int(self.entry_3.get())
                if (1.0 >= self.prob >= 0.0
                        and 1 < self.num < 40001
                        and 0 <= self.iso < 9):
                    self.tk.destroy()
                    break
                else:
                    raise ValueError('not a probability')

            except(ValueError):
                root = Tk()
                tkinter.messagebox.showinfo('Wrong input',
                                            'number of creatures should be an integer \n '
                                            'probability to infect should be a float \n '
                                            'probability must be between 0 and 1')
                root.destroy()
                root.mainloop()

    def run(self):
        """
        running the opening window.
        :return: the number, probability and isolation parameter.
        """
        self.widgets()
        self.tk.mainloop()
        return self.num, self.prob, self.iso


class CellularAutomata:
    """
    This class is responsible for the matrix that represents the CA grid.
    """
    def __init__(self, window_width, window_height, cell_size, n, p, k):
        """
        :param window_width: number of cells in width.
        :param window_height: number of cells in height.
        :param cell_size: size of a cell in pixels.
        :param n: size of the population.
        :param p: contamination probability.
        :param k: isolation factor.
        """
        self.grid_width = int(window_width / cell_size)
        self.grid_height = int(window_height / cell_size)
        self.cell_size = cell_size
        self.matrix = np.zeros((self.grid_width, self.grid_height), dtype=int) #create matrix of zeros
        self.generate_cells(n)
        self.move = Move(self.matrix, self.grid_width, self.grid_height, p, k)

    def get_grid_width(self):
        """
        :return: number of creatures width
        """
        return self.grid_width

    def get_grid_height(self):
        """
        :return: number of creatures height.
        """
        return self.grid_height

    def get_matrix(self):
        """
        :return: matrix representing cells.
        """
        return self.matrix

    def generate_cells(self, n):
        """
        This function will inhabit the matrix with N creatures randomly where one of them is sick.
        All the cells are started as 0, a healthy creature is represented by 1,
        a sick one by 2 isolation by .
        :param n: number of living creatures.
        :return: The matrix with N cells, 1 sick and N-1 healthy.
        """
        # creating random indexes for the matrix
        index = []
        for row in range(0, self.grid_width):
            for col in range(0, self.grid_height):
                index.append([row, col])
        # choose The indexes randomly using sample so we cant choose the same cell twice.
        population = rand.sample(index, k=n)
        # shuffle to randomly choose and leaving it in the first index to easy avoid.
        rand.shuffle(population)
        sick_index = population[0]
        # insert them into matrix.
        self.matrix[sick_index[0], sick_index[1]] = 2
        for item in population[1:]:
            self.matrix[item[0], item[1]] = 1
        return self.matrix

    def draw_grid(self):
        """
        drawing the grid - background grid lines.
        """
        pyglet.gl.glColor4f(0.23, 0.23, 0.23, 1.0)
        # Horizontal lines
        for i in range(0, self.grid_width):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', (0, i * self.cell_size, self.grid_width* self.cell_size, i * self.cell_size)))
        # Vertical lines
        for j in range(0, self.grid_height):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', (j * self.cell_size, 0, j * self.cell_size, self.grid_height*self.cell_size)))

    def rectangle(self, x1, y1, x2, y2, r, g, b):
        """
        upper left and bottom right position of the rectangle to draw.
        rgb for the color of the lines.
        :param x1: upper left
        :param y1: upper left
        :param x2: bottom right
        :param y2: bottom right
        :param r: red
        :param g: green
        :param b: blue
        """
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', (x1, y1, x1, y2, x2, y2, x2, y1)),
                             ('c3B', (r, g, b, r, g, b, r, g, b, r, g, b)))

    def draw_generation(self, steps):
        """
        draw generations
        :param steps: genetration number
        """
        label = pyglet.text.Label('Generation: %d' % steps,
                                  font_name='Times New Roman',
                                  font_size=16,
                                  x=300, y=610,
                                  anchor_x='center', anchor_y='center')
        label.draw()

    def draw(self):
        """
        drawing the population on the grid.
        """
        for row in range(0, self.grid_height):
            for col in range(0, self.grid_width):
                x1 = row * self.cell_size
                y1 = col * self.cell_size
                x2 = row * self.cell_size + self.cell_size
                y2 = col * self.cell_size + self.cell_size
                if self.matrix[row][col] == 1:
                    self.rectangle(x1, y1, x2, y2,
                                   153, 204, 255)
                if self.matrix[row][col] == 2:
                    self.rectangle(x1, y1, x2, y2,
                                   255, 0, 0)


class Move:
    def __init__(self, matrix, grid_width, grid_height, contamination_probability, isolation_k):
        """
        :param matrix:
        :param grid_width:
        :param grid_height:
        :param contamination_probability:
        :param isolation_k:
        """
        self.matrix = matrix
        self.p = contamination_probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.k = isolation_k

    def get_neighbours_indicies(self, row, col):
        """
        This method returns all the neighbours indexes of a given creature.
        :param row: row of the creature.
        :param col: column of the creature.
        :return: the indexes of all neighbours.
        """
        return [[(row - 1)%self.grid_width, (col - 1) % self.grid_height],
                     [row%self.grid_width, (col - 1) % self.grid_height],
                     [(row + 1) % self.grid_width, (col - 1) % self.grid_height],
                     [(row - 1) % self.grid_width, col % self.grid_height],
                     [(row + 1) % self.grid_width, col % self.grid_height],
                     [(row - 1) % self.grid_width, (col + 1) % self.grid_height],
                     [row % self.grid_width, (col + 1) % self.grid_height],
                     [(row + 1) % self.grid_width, (col + 1) % self.grid_height]]

    def contamination(self):
        """
        This method checks if a neighbour is sick and if it is the creature will become sick with a probability of p.
        isolation starts from ...
        :param k: isolation factor.
        :param p: probability to get sick
        :return: the tmp_matrix where the creature is now sick or not.
        """

        for row in range(0, self.grid_width):
            for col in range(0, self.grid_height):
                if self.matrix[row][col] == 2:
                    # check neighbours except those in isolation - first k neighbours in list.
                    for neighbour in self.get_neighbours_indicies(row, col)[self.k:]:
                        if self.matrix[neighbour[0]][neighbour[1]] == 1:
                            # each healthy neighbour has probability p to get sick
                            self.matrix[neighbour[0]][neighbour[1]] = \
                                np.random.choice([1, 2], size=1, replace=True, p=(1 - self.p, self.p))[0]

    def move(self):
        """
        creatures move.
        :return: the matrix with new positions.
        """
        for row in range(0, self.grid_width):
            for col in range(0, self.grid_height):
                if self.matrix[row][col] != 0:
                    tmp = self.matrix[row][col]
                    self.matrix[row][col] = 0
                    new_index = self.choose_index(row, col)
                    self.matrix[new_index[0]][new_index[1]] = tmp
        self.contamination()
        return self.matrix

    def choose_index(self, row, col):
        """
        This method chooses randomly where to move and returns that index.
        :param row: where the creature is now.
        :param col: where the creature is now.
        :return: the new index (or the same one) where the creature will be.
        """
        possible_neighbours = []
        for neighbour in self.get_neighbours_indicies(row, col):
            if self.matrix[neighbour[0]][neighbour[1]] == 0:
                possible_neighbours.append(neighbour)
        possible_neighbours.append([row, col])
        return random.choice(possible_neighbours)


class Window(pyglet.window.Window):
    def __init__(self, n, p, k):
        """
        :param n: number of creatures
        :param p: contamination probability
        :param k: isolation factor
        """
        super().__init__(WINDOW_SIZE[0], WINDOW_SIZE[1] + 20)
        self.num = n
        self.prob = p
        self.iso = k
        self.steps = 0
        self.ca = CellularAutomata(WINDOW_SIZE[0], WINDOW_SIZE[0], CELL_SIZE, self.num, self.prob, self.iso)
        # self.plot = Plotting(self.ca)  SHOULD HAVE BEEN ERASED
        pyglet.clock.schedule_interval(self.update, 1.0/1000000.0)

    def on_draw(self):
        """
        overriding base method in window- drawing on window.
        """
        self.clear()
        self.ca.draw()
        self.ca.draw_grid()
        self.ca.draw_generation(self.steps)

    def update(self, dt):
        """
        update matrix, move and steps after each generation.
        :param dt: needed for base method.
        """
        self.ca.move.move()
        self.steps += 1


if __name__ == "__main__":
    start_window = StartWindow()
    n, p, k = start_window.run()
    window = Window(n, p, k)
    pyglet.app.run()
