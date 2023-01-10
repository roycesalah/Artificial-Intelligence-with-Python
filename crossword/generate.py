import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.domains:
            deep_domain = set(self.domains[variable]).union({})
            for word in deep_domain:
                if variable.length != len(word):
                    self.domains[variable].remove(word)
        

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        match = self.crossword.overlaps[x,y]
        revised = False
        if match != None:
            i,j = match
            yj = []
            for yword in self.domains[y]:
                yj.append(yword[j])
            # Create temp domain which doesn't change size on iteration
            temp_domain = self.domains[x].union({})
            for xword in temp_domain:
                if xword[i] not in yj:
                    self.domains[x].remove(xword)
                    revised = True
        return revised

        

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Create list of all arcs in problem when default arcs = None
        if arcs == None:
            arcs = []
            for var in self.domains:
                for neighbor in self.crossword.neighbors(var):
                    temp_arc = set((var,neighbor))
                    if temp_arc not in arcs:
                        arcs.append(temp_arc)
        while len(arcs) != 0:
            x,y = tuple(arcs.pop(0))
            if self.revise(x,y):
                if len(self.domains[x]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor == y:
                        continue
                    arc = set([neighbor,x])
                    if arc not in arcs:
                        arcs.append(arc)
        return True

        


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.crossword.variables:
            if variable in assignment and assignment[variable] != None:
                continue
            else:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        consistent = True
        # Check if all distinct
        if len(set(assignment.keys())) != len(set(assignment.values())):
            return False
        # Check if word lengths are correct
        for variable in assignment:
            if variable.length != len(assignment[variable]):
                return False
        # Check if all are arc consistent
        for variable in assignment:
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment.keys():
                    overlap = self.crossword.overlaps[neighbor,variable]
                    if overlap != None:
                        i,j = overlap
                        if assignment[neighbor][i] != assignment[variable][j]:
                            return False
        return True

        raise NotImplementedError

    # Implement key function for order_domain_values sort
    def takeSecond(elem):
        return elem[1]

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Creates list of neighbors which aren't in assignment
        neighbors = list(set(self.crossword.neighbors(var)) - set(assignment.keys()))

        constraints = []
        # Iterate and add constraint count to list
        for word in self.domains[var]:
            count = 0
            for neighbor in neighbors:
                i,j = self.crossword.overlaps[var,neighbor]
                for neighbor_word in self.domains[neighbor]:
                    if word[i] != neighbor_word[j]:
                        count += 1
            constraints.append((word,count))
        
        # Sort list and create list of just words
        constraints.sort(key=self.takeSecond)
        sorted_domain = []
        for wordcount in constraints:
            sorted_domain.append(wordcount[0])
            
        return sorted_domain

        raise NotImplementedError

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # List of unassigned variables
        variables = list(self.domains.keys() - assignment.keys())
        
        # Determine lowest domain
        choice = []
        for var in variables:
            if len(choice) == 0:
                choice.append(var)
            elif len(self.domains[var]) == len(self.domains[choice[0]]):
                choice.append(var)
            elif len(self.domains[var]) < len(self.domains[choice[0]]):
                choice = [var]
        if len(choice) == 1:
            return choice[0]

        # Determine highest degree if multiple lowest domains
        choice_degree = []
        for var in choice:
            if len(choice_degree) == 0:
                choice_degree.append(var)
            elif len(self.crossword.neighbors(var)) == len(self.crossword.neighbors(choice_degree[0])):
                choice_degree.append(var)
            elif len(self.crossword.neighbors(var)) > len(self.crossword.neighbors(choice_degree[0])):
                choice_degree = [var]
        
        return choice_degree[0]

        raise NotImplementedError

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        test = assignment
        for value in self.domains[var]:
            assignment.update({var:value})
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result != False:
                    return result
            assignment.pop(var)
        return False


        raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
