import time
import numpy as np
import pandas as pd
from binClassNew import BuildingPlate
import csv


def placementProcedure(partsDict, nbParts, nbMachines, thresholds, chromosome, matching, collision_backend, plot=False):
    timeToFindPossibleBatch = 0
    timePlacingPart = 0

    SV = chromosome[:nbParts]
    MV = chromosome[nbParts:]

    worstMakespan = 0

    binsPerMachine = [[] for _ in range(nbMachines)]
    start = time.time()
    for i in range(nbMachines):
        if i == 0: ## Might have to reorganize these "if" statements because if I have more than 3 machines, it is more likely to fall under the "else"
            mask = MV <= thresholds[i]
        elif i == nbMachines - 1:
            mask = MV > thresholds[i-1]
        else:
            mask = (MV > thresholds[i-1]) & (MV <= thresholds[i])
        
        sequence = np.where(mask)[0]
        values = SV[sequence]

        sequence2 = matching[sequence]
        sorted_indices = np.argsort(values)
        sorted_sequence = sequence2[sorted_indices]
        #print(sorted_sequence)

        openBins = []
        for partInd in sorted_sequence:
            #print(partInd)
            result = False
            if (
                (partsDict[f'part{partInd}']['shapes0'][0] > partsDict[i]['binLength'] or partsDict[f'part{partInd}']['shapes0'][1] > partsDict[i]['binWidth'])
                and 
                (partsDict[f'part{partInd}']['shapes0'][1] > partsDict[i]['binLength'] or partsDict[f'part{partInd}']['shapes0'][0] > partsDict[i]['binWidth'])
                ):
                #print("INFEASIBLE Part: ",partInd, " Machine ", i+1)
                return 10000000000000000
            
            for l,bin in enumerate(openBins):
                
                # if it doesn't respect the area, skip to the next bin
                if bin.area + partsDict[f'part{partInd}']['area'] > partsDict[i]['binArea']: # add a check for length and width of the bounding boxes of the parts
                    continue

                
                # if it respects the area, check if it can be placed without overlap
                startFindPoss = time.time()
                result = bin.can_insert(partsDict[f'part{partInd}'], partsDict[i][f'part{partInd}'])
                timeToFindPossibleBatch += time.time()-startFindPoss
                #print("Time to find possible batch for part",partInd+1,"in batch",l,"is",time.time()-startFindPoss)
                if result:
                    break

            
            # In case all bins were checked and it was not possible to place the part, create a new batch
            startPlac = time.time()
            if not result:
                # Create a new bin, append it to the list of existing bins as well as to the machine that it belongs to
                newBin = BuildingPlate(partsDict[i]['binWidth'],partsDict[i]['binLength'], collision_backend)
                openBins.append(newBin)
                binsPerMachine[i].append(newBin)
                
                # Find the rotation that leads to a shorter bin length
                best_rotation = partsDict[f'part{partInd}']['lengths'].index(min(partsDict[f'part{partInd}']['lengths'])) # To avoid unnecessary computations I can do this when creating the part)
                
                # Insert the part in the bottomest-leftest position
                newBin.insert(0,partsDict[i]['binLength']-1, partsDict[f'part{partInd}'][f'rot{best_rotation}'], partsDict[f'part{partInd}'][f'shapes{best_rotation}'], partsDict[f'part{partInd}']['area'])
                
                # Update batch current state
                newBin.calculate_enclosure_box_length()  # Update box length
                #newBin.calculate_enclosure_box_width()  # Update box width
                
                newBin.processingTime += partsDict[i][f'part{partInd}']['procTime']
                newBin.processingTimeHeight = max(newBin.processingTimeHeight,partsDict[i][f'part{partInd}']['procTimeHeight'])
                newBin.partsAssigned.append(partsDict[f'part{partInd}']['id'])

                timePlacingPart += time.time()-startPlac


        
        #tpt = 0 
        makespan = 0
        for batch in openBins:
            #tpt += batch.resultFFT
            makespan += batch.processingTime + batch.processingTimeHeight + partsDict[i]['setupTime']
        # Open the CSV file in append mode
        '''file_path = "my_file.csv"  # Replace with your file path
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write the variable value in the next line
            writer.writerow([tpt])'''
        #print(tpt)
        if makespan > worstMakespan:
            worstMakespan = makespan

        if plot:
            for x, batch in enumerate(openBins):
                batch.save_plate_to_file(f"Final_Building_Plate_{i+1,x}.txt")
                print("Batch " ,x, " from machine " , i, batch.partsAssigned)
    #print(time.time()-start)
    #print("Time to find potential batches: ",timeToFindPossibleBatch)
    #print("Time to place the parts: ",timePlacingPart)
    #print("Total time: ", timeToFindPossibleBatch+timePlacingPart)
    #print("Total tardiness: ", totalTardiness)
    

    return worstMakespan
