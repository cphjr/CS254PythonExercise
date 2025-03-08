import random
def main():
    # Declare a one-dimensional array capable of storing 10 floating-point numbers
    float_array = [0.0] * 10

    #populate the array with random values within the range [-10.0, +10.0]
    for i in range(len(float_array)):
        float_array[i] = random.uniform(-10.0, 10.0)
        
    #Display all the elements of the array
    print(float_array)
    
if __name__ == "__main__":
    main()