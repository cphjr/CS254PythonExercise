import csv
import json

def load_csv(file_path):
    data = []
    with open(file_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append([float(value) for value in row])
    return data

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data['linear']

def main():
    # Load data from L1.csv
    l1_csv_data = load_csv("L1.csv")
    print(f"L1.csv data loaded with {len(l1_csv_data)} records.")

    # Load data from L1.json
    l1_json_data = load_json("L1.json")
    print(f"L1.json data loaded with {len(l1_json_data)} records.")

    # Load data from Q1.csv
    q1_csv_data = load_csv("Q1.csv")
    print(f"Q1.csv data loaded with {len(q1_csv_data)} records.")

    # Load data from Q2.csv
    q2_csv_data = load_csv("Q2.csv")
    print(f"Q2.csv data loaded with {len(q2_csv_data)} records.")

if __name__ == "__main__":
    main()