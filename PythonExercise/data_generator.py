import random
import json
import csv

def main():
    # Declare a two-dimensional array capable of storing 1,000,000 pairs of floating-point numbers (y,x)
    float_array = [[0.0, 0.0] for _ in range(1000000)]
    
    # Randomly generates x values in the range [-1000.0, +1000.0]
    for i in range(len(float_array)):
        x = random.uniform(-1000.0, 1000.0)
        y = 2.0 + 0.5 * x
        float_array[i] = [y, x]
        
    # Populate the array with the generated pairs and display the first, middle, and last pairs
    print(float_array[0])
    print(float_array[500000])
    print(float_array[999999])
    
    # Save the data to "L1.json"
    with open("L1.json", "w") as json_file:
        json.dump({"linear": float_array}, json_file)
    
    # Save the data to "L1.csv"
    with open("L1.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(float_array)

    # Declare another two-dimensional array capable of storing 1,000,000 pairs of floating-point numbers (y,x)
    float_array_2 = [[0.0, 0.0] for _ in range(1000000)]
    
    # Randomly generates x values in the range [-1000.0, +1000.0]
    for i in range(len(float_array_2)):
        x = random.uniform(-1000.0, 1000.0)
        y = 2.0 + 0.5 * x - 3 * x ** 2
        float_array_2[i] = [y, x]
        
    # Populate the array with the generated pairs and display the first, middle, and last pairs
    print(float_array_2[0])
    print(float_array_2[500000])
    print(float_array_2[999999])
    
    # Save the data to "Q1.csv"
    with open("Q1.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(float_array_2)
    
    # Save the data to "Q2.csv" with three columns (y, x1, x2)
    with open("Q2.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for y, x in float_array_2:
            x1 = 0.5 * x
            x2 = -3 * x ** 2
            writer.writerow([y, x1, x2])

if __name__ == "__main__":
    main()