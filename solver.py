import numpy as np
import math
import argparse

class TileShape:
    # TileShape class intended to be immutable

    def __init__(self, tileshape: np.ndarray):
        # tileshape is a bool ndarray with True at the cells filled in by the tile
        self.grid = tileshape
        self.hash_cache = None

    def flip(self) -> 'TileShape':
        """Mirrors across Y axis, returns new TileShape."""
        return TileShape(np.flip(self.grid, 1))

    def rotate(self, n: int) -> 'TileShape':
        """Rotates clockwise by N 90 degree rotations."""
        n = n % 4
        if n > 0:
            return TileShape(np.rot90(self.grid, k=n, axes=(1, 0)))

    def __eq__(self, other: 'TileShape') -> bool:
        return np.array_equal(self.grid, other.grid)

    def __ne__(self, other: 'TileShape') -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        if self.hash_cache is not None:
            return self.hash_cache
        self.hash_cache = hash(self.grid.data.tobytes())
        return self.hash_cache

    def visual_str(self) -> str:
        """Returns a multiline string with a visual representation of the shape"""
        full_char = 'X'
        empty_char = ' '
        r = ''
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                r += full_char if self.grid[y, x] else empty_char
            r += '\n'
        return r


class GridState:

    def __init__(self, copy_other = None):
        self.hash_cache = None
        if copy_other:
            self.grid = copy_other.grid.copy()
            self.num_empty_cells = copy_other.num_empty_cells
        else:
            # The grid of dates, where each cell is a boolean whether the cell is occupied or not.
            # The grid is indexed as self.grid[y, x]
            self.grid: np.ndarray = np.full((7, 7), False, dtype=np.bool_)
            # Fill in the initial unusable cells (y, x)
            unusable_cells: list[tuple[int, int]] = [ (0, 6), (1, 6), (6, 3), (6, 4), (6, 5), (6, 6) ]
            # Count of empty cells
            self.num_empty_cells: int = self.grid.shape[0] * self.grid.shape[1]
            for coord in unusable_cells:
                self.fill_cell(coord)

    def fill_cell(self, coord: tuple[int, int], ignore_full: bool = False) -> None:
        # coord is (y, x)
        assert coord[0] >= 0 and coord[0] < self.grid.shape[0]
        assert coord[1] >= 0 and coord[1] < self.grid.shape[1]
        if not ignore_full:
            assert self.grid[coord] == False
        if not self.grid[coord]:
            self.num_empty_cells -= 1
        self.grid[coord] = True

    def fill_shape(self, shape: TileShape, offset: tuple[int, int], ignore_full: bool = False) -> None:
        # offset is (y, x)
        for y in range(shape.grid.shape[0]):
            for x in range(shape.grid.shape[1]):
                if shape.grid[y, x]:
                    self.fill_cell((y + offset[0], x + offset[1]), ignore_full)

    def can_fit(self, shape: TileShape, offset: tuple[int, int]) -> bool:
        # offset is (y, x)
        for y in range(shape.grid.shape[0]):
            for x in range(shape.grid.shape[1]):
                if shape.grid[y, x] and self.grid[y + offset[0], x + offset[1]]:
                    return False
        return True

    def copy(self) -> 'GridState':
        return GridState(self)

    def __eq__(self, other: 'GridState') -> bool:
        return np.array_equal(self.grid, other.grid)

    def __ne__(self, other: 'GridState') -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        if self.hash_cache is not None:
            return self.hash_cache
        self.hash_cache = hash(self.grid.data.tobytes())
        return self.hash_cache



class Solver:

    def __init__(self):
        # The shapes of each of the possible tiles.
        base_shapes: list[TileShape] = [
            TileShape(np.array([ [ True, True, True ], [ True, True, True ] ], dtype=np.bool_)), # 3x2 rectangle
            TileShape(np.array([ [ True, False, True ], [ True, True, True ] ], dtype=np.bool_)), # 3x2 U shape
            TileShape(np.array([ [ False, True, True ], [ False, True, False ], [ True, True, False ] ], dtype=np.bool_)), # 3x3 S shape
            TileShape(np.array([ [ True, False, False ], [ True, False, False ], [ True, True, True ] ], dtype=np.bool_)), # 3x3 L shape
            TileShape(np.array([ [ True, False ], [ True, False ], [ True, False ], [ True, True ] ], dtype=np.bool_)), # 2x4 L shape
            TileShape(np.array([ [ False, False, True, True ], [ True, True, True, False ] ], dtype=np.bool_)), # 4x2 S shape
            TileShape(np.array([ [ True, False ], [ True, True ], [ True, True ] ], dtype=np.bool_)), # 2x3 almost-rectangle
            TileShape(np.array([ [ False, True, False, False ], [ True, True, True, True ] ], dtype=np.bool_)) # 4x2 "bump on log" shape
        ]
        # Convert each tile shape into a set of possible orientations (combinations of rotations and mirrors)
        def get_all_shape_orientations(shape: TileShape):
            shapeset: TileShape = set([ # dedup using a set
                shape,
                shape.rotate(1),
                shape.rotate(2),
                shape.rotate(3),
                shape.flip(),
                shape.flip().rotate(1),
                shape.flip().rotate(2),
                shape.flip().rotate(3)
            ])
            return list(shapeset)
        self.shape_orientations: list[list[TileShape]] = [ get_all_shape_orientations(shape) for shape in base_shapes ]
        # Will be assigned to a function used to check for solution.
        # The function has the signature (GridState) -> int; where the return value can be 0 (not solved), 1 (solved), or 2 (not solved, and unsolvable with additional pieces)
        self.check_solution = None
        # Will be filled in when a solution is found
        self.solution_tile_offsets = None
        # Used for optimization
        self.visited_states: GridState = set()

    def set_solution_empty_cells(self, coords: list[tuple[int, int]]) -> None:
        """Sets the solution condition to be a specific set of empty cells."""
        # coords are (y, x)
        def check_solution(state: GridState):
            if state.num_empty_cells < len(coords):
                return 2
            for y, x in coords:
                if state.grid[y, x]:
                    return 2
            if state.num_empty_cells == len(coords):
                return 1
            else:
                return 0
        self.check_solution = check_solution

    def run_solve(self):
        def check_tile(state: GridState, tilenum: int, tile_offsets: list[tuple[int, int, int]]) -> bool:
            if tilenum >= len(self.shape_orientations):
                return False
            if state in self.visited_states:
                return False
            self.visited_states.add(state)
            # Check each potential orientation and position of this tile
            for orientnum, orient in enumerate(self.shape_orientations[tilenum]):
                for offset_y in range(state.grid.shape[0] - orient.grid.shape[0] + 1):
                    for offset_x in range(state.grid.shape[1] - orient.grid.shape[1] + 1):
                        if state.can_fit(orient, (offset_y, offset_x)):
                            newstate = state.copy()
                            newstate.fill_shape(orient, (offset_y, offset_x))
                            s = self.check_solution(newstate)
                            if s == 1: # solved
                                self.solution_tile_offsets = tile_offsets + [ (offset_y, offset_x, orientnum) ]
                                return True
                            elif s == 0:
                                newoffsets = tile_offsets.copy()
                                newoffsets.append(( offset_y, offset_x, orientnum ))
                                r = check_tile(newstate, tilenum + 1, newoffsets)
                                if r:
                                    return True
        r = check_tile(GridState(), 0, [])
        assert r, 'Solution not found'

    def solution_str(self) -> str:
        """String representation of the solution (multiline)."""
        gridstr = ''
        sgrid = np.full(GridState().grid.shape, 'X', dtype=object)
        for tilenum, (off_y, off_x, orientnum) in enumerate(self.solution_tile_offsets):
            orient = self.shape_orientations[tilenum][orientnum]
            for tile_y in range(orient.grid.shape[0]):
                for tile_x in range(orient.grid.shape[1]):
                    if orient.grid[tile_y, tile_x]:
                        sgrid[off_y + tile_y, off_x + tile_x] = str(tilenum)
        for y in range(sgrid.shape[0]):
            for x in range(sgrid.shape[1]):
                gridstr += sgrid[y, x]
            gridstr += '\n'
        return gridstr


    def print_solution(self):
        for tilenum, (y, x, orientnum) in enumerate(self.solution_tile_offsets):
            print(f'Tile #{tilenum} in orientation #{orientnum} goes into position (x, y) = ({x}, {y}):')
            print(self.shape_orientations[tilenum][orientnum].visual_str())
            print()
        print('Solved grid:')
        sgrid = np.full(GridState().grid.shape, 'X', dtype=object)
        for tilenum, (off_y, off_x, orientnum) in enumerate(self.solution_tile_offsets):
            orient = self.shape_orientations[tilenum][orientnum]
            for tile_y in range(orient.grid.shape[0]):
                for tile_x in range(orient.grid.shape[1]):
                    if orient.grid[tile_y, tile_x]:
                        sgrid[off_y + tile_y, off_x + tile_x] = str(tilenum)
        print(self.solution_str())


def solve_for_date(month, day):
    """Runs and displays solver output for given date.  Month and day parameters are 1-indexed."""
    # Convert month and day to 0 indexed internally
    month -= 1
    day -= 1
    month_cell_y = 0 if month < 6 else 1
    month_cell_x = month - month_cell_y * 6
    day_cell_y = 2 + math.floor(day / 7)
    day_cell_x = day % 7
    solver = Solver()
    solver.set_solution_empty_cells([ ( month_cell_y, month_cell_x ), ( day_cell_y, day_cell_x ) ])
    solver.run_solve()
    return solver


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Date calendar tile puzzle solver')
    argparser.add_argument('month', type=int, help='Month number starting at 1')
    argparser.add_argument('day', type=int, help='Day number starting at 1')
    args = argparser.parse_args()
    solver = solve_for_date(args.month, args.day)
    solver.print_solution()





