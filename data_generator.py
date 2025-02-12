import random
import json
import csv

def main():
    # Declare a two-dimensional array capable of storing 100,000,000 pairs of floating-point numbers (y,x)
    float_array = [[0.0, 0.0] for _ in range(100000000)]
    
    # Randomly generates x values in the range [-1000.0, +1000.0]
    for i in range(len(float_array)):
        float_array[i][0] = random.uniform(-1000.0, 1000.0)
        # Computes the corresponding y values using the equation y = 2.0 + 0.5x
        float_array[i][1] = 2.0 + 0.5 * float_array[i][0]
        
    # Populate the array with the generated pairs and display the first, middle, and last pairs
    print(float_array[0])
    print(float_array[50000000])
    print(float_array[99999999])
    
    # Save the data to "L1.json"
    with open("L1.json", "w") as json_file:
        json.dump({"linear": float_array}, json_file)
    
    # Save the data to "L1.csv"
    with open("L1.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(float_array)

if __name__ == "__main__":
    main()