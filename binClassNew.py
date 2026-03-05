import numpy as np
import time
from collision_backend import create_collision_backend

class BuildingPlate:
    def __init__(self, width, length, collision_backend=None):
        #### NESTING CHARACTERISTICS ####
        self.width = width
        self.length = length
        self.enclosure_box_length = 0
        self.enclosure_box_width = 0
        self.area = 0
        # Initialize the first representation of the grid with zeros (no parts placed yet)
        self.grid = np.zeros((length, width), dtype=int)
        # Initialize the second representation of the grid with zeros (no parts placed yet)
        self.grid2 = np.zeros((length, width), dtype=int)
        # Initialize Vacancy Vector (VV) with maximum possible vacancy
        self.vacancy_vector = np.zeros(length, dtype=int) + width

        #### SCHEDULING CHARACTERISTICS ####
        self.processingTime = 0
        self.processingTimeHeight = 0
        self.partsAssigned = []

        self.resultFFT = 0
        self.collision_backend = collision_backend or create_collision_backend("torch_gpu")
        self.grid_state = self.collision_backend.create_grid_state(length, width)


    def save_plate_to_file(self, filename):
        with open(filename, 'w') as file:
            for row in self.grid:
                file.write(' '.join(f'{val:2d}' for val in row) + '\n')

    #@profile
    def can_insert(self, part, machPart, plott=False):
        result = False
        best_pixel, best_rotation, packingDensity = None, 0, 0 # Initialize packing density at zero  
        potentialArea = (self.area+part['area'])

        # Get tensor from current binary grid
        startTim = time.time()

        feasible_rotations = []
        feasible_shapes = []
        feasible_ffts = []

        for currRot in range(part['nrot']):
            subarrays = np.lib.stride_tricks.sliding_window_view(self.vacancy_vector, part[f'shapes{currRot}'][0])
            binaryResult = np.any(np.all(subarrays >= part[f'dens{currRot}'], axis=1))
            
            if binaryResult:
                feasible_rotations.append(currRot)
                feasible_shapes.append(part[f'shapes{currRot}'])
                feasible_ffts.append(machPart[f'fft{currRot}'])

        batch_results = self.collision_backend.find_bottom_left_zero_batch(
            self.grid,
            feasible_ffts,
            feasible_shapes,
            grid_state=self.grid_state,
        )
        for i, currRot in enumerate(feasible_rotations):
            feasible, smallest_col_with_zero, largest_row_with_zero_real_value = batch_results[i]
            if feasible:
                result = True
                largest_row_with_zero = largest_row_with_zero_real_value - part[f'shapes{currRot}'][0] + 1

                newLength = max(self.enclosure_box_length, self.length - largest_row_with_zero)
                newPackingDensity = potentialArea/(newLength*self.width)

                if  newPackingDensity > packingDensity:
                    best_pixel, best_rotation, packingDensity = [smallest_col_with_zero,largest_row_with_zero_real_value], currRot, newPackingDensity

                elif newPackingDensity == packingDensity and largest_row_with_zero_real_value > best_pixel[1]:
                    best_pixel, best_rotation, packingDensity = [smallest_col_with_zero,largest_row_with_zero_real_value], currRot, newPackingDensity
                
                elif newPackingDensity == packingDensity and largest_row_with_zero_real_value == best_pixel[1] and smallest_col_with_zero < best_pixel[0]:
                    best_pixel, best_rotation, packingDensity = [smallest_col_with_zero,largest_row_with_zero_real_value], currRot, newPackingDensity
        self.resultFFT += time.time() - startTim
        
        if result == True:
            # Insert the part in the best pixel found
            self.insert(best_pixel[0], best_pixel[1], part[f'rot{best_rotation}'], part[f'shapes{best_rotation}'], part['area'])
            self.calculate_enclosure_box_length()  # Update box length

            self.partsAssigned.append(part['id'])
            
            self.processingTime += machPart['procTime']
            self.processingTimeHeight = max(self.processingTimeHeight,machPart['procTimeHeight'])

            '''if plott:
                with open('output.txt', "a") as file:
                    file.write("\n")
                    file.write(f"Part {part['id']} placed in rotation {best_rotation}")'''
            
            return result
        
        return result
    
    
    def calculate_enclosure_box_length(self):
        # Find the first row index with at least one 1
        first_row_with_one = np.where(self.grid.any(axis=1))[0][0]
        self.enclosure_box_length = self.length - first_row_with_one

    
    def insert(self, x, y, partMatrix, shapes, partArea):
        self.area += partArea
        # Use slicing to insert the binary part matrix
        self.grid[y - shapes[0] + 1:y + 1, x:x + shapes[1]] += partMatrix
        self.collision_backend.update_grid_region(self.grid_state, x, y, partMatrix, shapes)

        # Do the required changes on the vacancy vector
        # Step 1: Pad the matrix with ones
        binaryGrid = self.grid[y - shapes[0] + 1:y + 1, :]
        padded_matrix = np.pad(binaryGrid, ((0, 0), (1, 1)), constant_values=1)

        # Step 2: Compute differences
        diffs = np.diff(padded_matrix, axis=1)

        # Step 3: Identify start and end of zero runs
        start_indices = np.where(diffs == -1)
        end_indices = np.where(diffs == 1)

        # Step 4: Compute lengths of zero runs
        run_lengths = end_indices[1] - start_indices[1]

        # Step 5: Create a result array initialized with zeros
        max_zeros = np.zeros(binaryGrid.shape[0], dtype=int)

        # Step 6: Use np.maximum.at to find the maximum run length for each row
        np.maximum.at(max_zeros, start_indices[0], run_lengths)

        # Step 7: Update the vacancy vector
        self.vacancy_vector[y - shapes[0] + 1:y + 1] = max_zeros
            
