import math
import random
import numpy as np
from placement import placementProcedure
from binClassInitialSol import BuildingPlate
from concurrent.futures import ThreadPoolExecutor
import time 
import sys
import torch
import itertools
import pandas as pd
from collision_backend import create_collision_backend
torch.set_num_threads(1)

class BRKGA():
    def __init__(self, Parts, nbParts, nbMachines, thresholds, instanceParts, initialSol,
                 collision_backend, num_generations = 200, num_individuals=100, num_elites = 12, num_mutants = 18, eliteCProb = 0.7):

        # Input
        self.Parts = Parts
        self.nbMachines = nbMachines
        self.thresholds = thresholds
        self.N = nbParts
        self.instanceParts = instanceParts
        self.initialSol = initialSol
        self.collision_backend = collision_backend
        
        # Configuration
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        
        # Result
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {
            'mean': [],
            'min': [],
            'time': []
        }


    def evaluate_solution(self, solution):
        decoder = placementProcedure(
            self.Parts,
            self.N,
            self.nbMachines,
            self.thresholds,
            solution,
            self.instanceParts,
            self.collision_backend,
        )
        return decoder

    def cal_fitness(self, population):
        with ThreadPoolExecutor(max_workers=4) as executor:
            fitness_list = list(executor.map(self.evaluate_solution, population))
        return fitness_list

    def partition(self, population, fitness_list):
        sorted_indexs = np.argsort(fitness_list)
        return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]], np.array(fitness_list)[sorted_indexs[:self.num_elites]]
    
    def crossover(self, elite, non_elite):
        # Generate random probabilities for each gene and create a boolean mask where True indicates the gene should come from elite
        crossover_mask = np.random.uniform(low=0.0, high=1.0, size=self.num_gene) < self.eliteCProb
        # Use the mask to choose genes from elite and non_elite
        return np.where(crossover_mask, elite, non_elite).tolist()
    
    def mating(self, elites, non_elites):
        # biased selection of mating parents: 1 elite & 1 non_elite
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return [self.crossover(random.choice(elites), random.choice(non_elites)) for i in range(num_offspring)]
    
    def mutants(self):
        return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))
        
    def fit(self, verbose = False):
        startfit = time.time()
        # Initial population & fitness
        population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
        population[0] = self.initialSol
        fitness_list = self.cal_fitness(population)
        
        if verbose:
            print('\nInitial Population:')
            print('  ->  shape:',population.shape)
            print('  ->  Best Fitness:',min(fitness_list))
            
        # best    
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))
        self.history['time'].append(time.time()-startfit)
        
        
        # Repeat generations
        best_iter = 0
        
        for g in range(self.num_generations):
            startTime = time.time()
            # Select elite group
            elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
            print(np.average(elite_fitness_list))
            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)
            
            # Generate mutants
            mutants = self.mutants()

            # New Population & fitness
            offspring = np.concatenate((mutants,offsprings), axis=0)
            offspring_fitness_list = self.cal_fitness(offspring)
            
            population = np.concatenate((elites, mutants, offsprings), axis = 0)
            fitness_list = list(elite_fitness_list) + offspring_fitness_list

            # Update Best Fitness
            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]
            
            self.history['min'].append(np.min(fitness_list))
            self.history['mean'].append(np.mean(fitness_list))
            self.history['time'].append(time.time()-startfit)
            
            if verbose:
                print("Generation :", g, ' \t(Best Fitness:', best_fitness,')')
            
            print(time.time()-startTime)

        self.used_bins = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution
        return 'feasible'
    
if __name__ == "__main__":
    '''INITIAL AND KNOWN DATA'''
    nbParts = int(sys.argv[1])
    #nbParts = 25
    nbMachines = int(sys.argv[2])
    #nbMachines = 2
    instNumber = int(sys.argv[3])
    #instNumber = 0
    backend_name = sys.argv[4] if len(sys.argv) > 4 else "torch_gpu"
    collision_backend = create_collision_backend(backend_name)

    '''DEFINE DATA'''
    with open(f'data/Instances/P{nbParts}M{nbMachines}-{instNumber}.txt', 'r') as file:
        data = file.read()
    # read instance to know which parts are in it
    instanceParts = np.array([int(x) for x in data.split()])
    #instanceParts = np.array([38,38])
    instancePartsUnique = np.unique(instanceParts)
    
    
    # Job specifications
    jobSpecAll = pd.read_excel(f'data/PartsMachines/part-machine-information.xlsx', sheet_name='part', header = 0, index_col = 0)
    jobSpec = jobSpecAll.loc[instancePartsUnique]
    #print(jobSpec)
    # Machine specifications
    machSpec = pd.read_excel(f'data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header = 0, index_col = 0)
    
    # Area of each job
    area = pd.read_excel(f'data/PartsMachines/polygon_areas.xlsx', header = 0)["Area"].tolist()
    
    # Rotations of each job
    polRotations = pd.read_excel(f'data/PartsMachines/parts_rotations.xlsx', header = 0)["rot"].tolist()

    data = {}
    # Load the binary matrices from .npy files
    startLoad = time.time()
    parts = []
    for m in range(nbMachines):
        data[m] = {}
        binLength = machSpec['L(mm)'].iloc[m]
        binWidth = machSpec['W(mm)'].iloc[m]
        binHeight = machSpec['H(mm)'].iloc[m]
        binArea = binLength*binWidth
        data[m]['binLength'] = binLength
        data[m]['binWidth'] = binWidth
        data[m]['binArea'] = binArea
        data[m]['setupTime'] = machSpec['ST(s)'].iloc[m]
        binVolume = binLength*binWidth*binHeight
        for part in instancePartsUnique:
            matrix = np.load(f'data/partsMatrices/matrix_{part}.npy')
            matrix = matrix.astype(np.int32)
            matrix = np.ascontiguousarray(matrix)
            if np.array_equal(matrix,np.rot90(matrix,2)):
                nrot = 2
            else:
                nrot = 4
            #nrot = polRotations[part]
            
            data[f'part{part}'] = {}
            data[m][f'part{part}'] = {}
            for rot in range(nrot):
                data[f'part{part}'][f'rot{rot}'] = np.rot90(matrix,rot)
                data[f'part{part}'][f'dens{rot}'] = np.array([max(len(list(g)) for k, g in itertools.groupby(row) if k) for row in data[f'part{part}'][f'rot{rot}']])
                data[f'part{part}'][f'shapes{rot}'] = [data[f'part{part}'][f'rot{rot}'].shape[0],data[f'part{part}'][f'rot{rot}'].shape[1]]

                data[m][f'part{part}'][f'fft{rot}'] = collision_backend.prepare_part_fft(
                    data[f'part{part}'][f'rot{rot}'],
                    binLength,
                    binWidth,
                )

            data[f'part{part}']['area'] = area[part]
            #data[f'part{part}']['len'] = jobSpec["length(mm)"].loc[part]
            #data[f'part{part}']['wid'] = jobSpec["width(mm)"].loc[part]
            
            
            data[m][f'part{part}']['procTime'] = jobSpec["volume(mm3)"].loc[part]*machSpec["VT(s/mm3)"].iloc[m] + jobSpec["support(mm3)"].loc[part]*machSpec["SPT(s/mm3)"].iloc[m]
            data[m][f'part{part}']['procTimeHeight'] = jobSpec["height(mm)"].loc[part]*machSpec["HT(s/mm3)"].iloc[m]
            
            data[f'part{part}']['nrot'] = nrot
            data[f'part{part}']['id'] = part

            data[f'part{part}']['lengths'] = [data[f'part{part}'][f'shapes{currRot}'][0] for currRot in range(nrot)]

    #print(time.time()-startLoad)
    thresholds = [t / nbMachines for t in range(1, nbMachines)] # define the the thresholds for the random keys of the BRKGA for machine assignment


    ''' CREATE INITIAL SOLUTION '''
    # Create dictionary that will hold info about which parts are assgined to which machines and their sequence
    machines_dict = {f'machine_{i}': {'makespan': 0, 'parts':[], 'batches':[]} for i in range(nbMachines)}
    # Order parts by decreasing order of height, breaking ties by descending order of projection area
    partsInfo = jobSpec.loc[instanceParts]["height(mm)"]
    partsAR = pd.read_excel(f'data/PartsMachines/polygon_areas.xlsx', header = 0, index_col = 0).loc[instanceParts]["Area"]
    conc = pd.concat([partsInfo, partsAR], axis = 1)
    sorted_df = conc.sort_values(by=['height(mm)', 'Area'], ascending=[False, False])
    part_sortedSequence = sorted_df.index.to_list()

    for part in part_sortedSequence:
        # Variable to hold makespan of the system
        best_makespan = 1000000000000000000
        bestBatch = []
        
        for mach in range(nbMachines):
            placedInExist = False
            if (
                (data[f'part{part}']['shapes0'][0] > data[mach]['binLength'] or data[f'part{part}']['shapes0'][1] > data[mach]['binWidth'])
                and 
                (data[f'part{part}']['shapes0'][1] > data[mach]['binLength'] or data[f'part{part}']['shapes0'][0] > data[mach]['binWidth'])
                ):
                continue
            
            machineMakespan = machines_dict[f'machine_{mach}']['makespan'] + data[mach]['setupTime'] + data[mach][f'part{part}']['procTime'] + data[mach][f'part{part}']['procTimeHeight']
            newMakespan = max(max([machines_dict[f'machine_{i}']['makespan'] for i in range(nbMachines)]), machineMakespan)
            #print("Place part ", part, " in a new batch in machine ", mach, "leads to makespan ", newMakespan)


            if newMakespan <= best_makespan:
                best_makespan = newMakespan
                # Find the rotation that leads to a shorter bin length
                best_rotation = data[f'part{part}']['lengths'].index(min(data[f'part{part}']['lengths'])) # To avoid unnecessary computations I can do this when creating the part)
                bestBatch = ['new', mach, [0,data[mach]['binLength']-1], best_rotation, machineMakespan]
            
            for x,batch in enumerate(machines_dict[f'machine_{mach}']['batches']):
                res = batch.can_insert(data[f'part{part}'], data[mach][f'part{part}'])
                if res[0]:
                    if batch.processingTimeHeight < data[mach][f'part{part}']['procTimeHeight']:
                        machineMakespan = machines_dict[f'machine_{mach}']['makespan'] - batch.ProcessingTimeHeight + data[mach][f'part{part}']['procTimeHeight'] + data[mach][f'part{part}']['procTime']
                    else:
                        machineMakespan = machines_dict[f'machine_{mach}']['makespan'] + data[mach][f'part{part}']['procTime']

                    newMakespan = max(max([machines_dict[f'machine_{i}']['makespan'] for i in range(nbMachines)]), machineMakespan)
                    #print("Place part ", part, " in batch", x, "in machine ", mach, "leads to makespan ", newMakespan)
                    if newMakespan <= best_makespan:
                        if placedInExist and newMakespan == best_makespan:
                            continue
                        else:
                            placedInExist = True
                            best_makespan = newMakespan
                            bestBatch = ['exist', mach, res[1], res[2], machineMakespan, batch]

        if bestBatch[0] == 'new':
            #print("Placed part ",part," in a new batch in machine ", bestBatch[1]," and current makespan is ", bestBatch[4])
            machines_dict[f'machine_{bestBatch[1]}']['parts'].append(part)
            machines_dict[f'machine_{bestBatch[1]}']['makespan'] = bestBatch[4]
            newBin = BuildingPlate(data[bestBatch[1]]['binWidth'],data[bestBatch[1]]['binLength'], collision_backend)
            machines_dict[f'machine_{bestBatch[1]}']['batches'].append(newBin)
            
            # Insert the part in the bottomest-leftest position
            newBin.insert(0,data[bestBatch[1]]['binLength']-1, data[f'part{part}'][f'rot{bestBatch[3]}'], data[f'part{part}'][f'shapes{bestBatch[3]}'], data[f'part{part}']['area'])
            
            # Update batch current state
            newBin.calculate_enclosure_box_length()  # Update box length
            #newBin.calculate_enclosure_box_width()  # Update box width
            
            newBin.processingTime += data[bestBatch[1]][f'part{part}']['procTime']
            newBin.processingTimeHeight = max(newBin.processingTimeHeight,data[bestBatch[1]][f'part{part}']['procTimeHeight'])
            newBin.partsAssigned.append(data[f'part{part}']['id'])
            
        elif bestBatch[0] == 'exist':
            #print("Placed part ",part," in an existing batch in machine ", bestBatch[1]," and current makespan is ", bestBatch[4])
            machines_dict[f'machine_{bestBatch[1]}']['parts'].append(part)
            machines_dict[f'machine_{bestBatch[1]}']['makespan'] = bestBatch[4]
            newBin = bestBatch[5]
            
            # Find the rotation that leads to a shorter bin length
            best_rotation = data[f'part{part}']['lengths'].index(min(data[f'part{part}']['lengths'])) # To avoid unnecessary computations I can do this when creating the part)
            
            # Insert the part in the bottomest-leftest position
            newBin.insert(bestBatch[2][0],bestBatch[2][1], data[f'part{part}'][f'rot{bestBatch[3]}'], data[f'part{part}'][f'shapes{bestBatch[3]}'], data[f'part{part}']['area'])
            
            # Update batch current state
            newBin.calculate_enclosure_box_length()  # Update box length
            #newBin.calculate_enclosure_box_width()  # Update box width
            
            newBin.processingTime += data[bestBatch[1]][f'part{part}']['procTime']
            newBin.processingTimeHeight = max(newBin.processingTimeHeight,data[bestBatch[1]][f'part{part}']['procTimeHeight'])
            newBin.partsAssigned.append(data[f'part{part}']['id'])
    
    array = np.zeros(2*nbParts)

    used_indices = set()
    for m in range(nbMachines):
        positions = []
        
        partsMachine = np.concatenate([batch.partsAssigned for batch in machines_dict[f'machine_{m}']['batches']])
        for value in partsMachine:
            for idx, val in enumerate(instanceParts):
                if val == value and idx not in used_indices:
                    positions.append(idx)
                    used_indices.add(idx)
                    break
        
        positions_array = np.array(positions)

        if m == 0: ## Might have to reorganize these "if" statements because if I have more than 3 machines, it is more likely to fall under the "else"
            array[positions_array+nbParts] = random.uniform(0,thresholds[m])    
            #mask = MV <= thresholds[i]
        elif m == nbMachines - 1:
            array[positions_array+nbParts] = random.uniform(thresholds[m-1]+0.0001,1-0.0001)    
            #mask = MV > thresholds[i-1]
        else:
            array[positions_array+nbParts] = random.uniform(thresholds[m-1]+0.0001,thresholds[m]-0.0001)
            #mask = (MV > thresholds[i-1]) & (MV <= thresholds[i])

        # Generate strictly increasing values between 0 and 1
        values = np.linspace(0, 1, len(positions_array), endpoint=False)  # Equally spaced values

        # Assign these values to the specified positions in arrayZeros
        for i, pos in enumerate(positions_array):
            array[pos] = values[i]
    
    print("Makespan of initial solution: ",best_makespan)

    ''' CALL BRKGA'''
    # Possible values for p
    #prob = [10,15,20,30]
    prob = [10]
    
    for mult in prob:
        # Initialize the Excel writer object
        name = f'P{nbParts}M{nbMachines}-{instNumber}'
        writer = pd.ExcelWriter(f'OriginalInitialSol_{name}_prob_{mult}.xlsx', engine='openpyxl')
        for i in range(1):
            model = BRKGA(data, nbParts, nbMachines, thresholds, instanceParts, array,
                  collision_backend=collision_backend,
                  num_generations = 30, num_individuals=mult*nbParts, num_elites = math.ceil(mult*nbParts*0.1), num_mutants = math.ceil(mult*nbParts*0.15), eliteCProb = 0.70)
            model.fit(verbose = True)

            # Convert dictionary to DataFrame
            df = pd.DataFrame(model.history)
            # Export DataFrame to Excel file
            df.to_excel(writer, sheet_name = f'Iteration{i+1}', index=False)
            
            del model

        # Save the Excel file
        writer.close()
