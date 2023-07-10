<<<<<<< HEAD
import csv
import os
import random
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='mutation status csv')
    parser.add_argument('--testpercent', type=float, help='local test verbose')
    parser.add_argument('--trainname', type = str, help = 'new training csv file name')
    parser.add_argument('--testname', type = str, help = 'new testing csv file name')
    args = parser.parse_args()
    train_csv, test_csv = split_csv(args)

    return train_csv, test_csv


def split_csv(args):
    # Create output directories
    output_dir = os.path.dirname(args.file)
    train_dir = os.path.join(output_dir, 'train')


    # Copy the input CSV file to train and test directories
    train_csv = os.path.join(output_dir, args.trainname)
    test_csv = os.path.join(output_dir, args.testname)
    shutil.copyfile(args.file, train_csv)
    shutil.copyfile(args.file, test_csv)

    # Calculate the number of testing rows
    total_rows = sum(1 for line in open(args.file)) - 1  # Subtract 1 for header row
    testing_rows = int(total_rows * args.testpercent)

    # Shuffle and split the rows
    with open(args.file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        data = list(reader)
        random.shuffle(data)

        # Write testing rows to testing.csv
        with open(test_csv, 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerow(header)
            writer.writerows(data[:testing_rows])

        # Write remaining rows to training.csv
        with open(train_csv, 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(header)
            writer.writerows(data[testing_rows:])

    return train_csv, test_csv


if __name__ == '__main__':
    main()

# # Example usage
# csv_file = './mutation/BTF_all.csv'
# train_csv, test_csv = split_csv(csv_file, 0.2)  # 20% of data for testing
#
# print(f"Training CSV: {train_csv}")
# print(f"Testing CSV: {test_csv}")


=======
import csv
import os
import random
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='mutation status csv')
    parser.add_argument('--testpercent', type=float, help='local test verbose')
    parser.add_argument('--trainname', type = str, help = 'new training csv file name')
    parser.add_argument('--testname', type = str, help = 'new testing csv file name')
    args = parser.parse_args()
    train_csv, test_csv = split_csv(args)

    return train_csv, test_csv


def split_csv(args):
    # Create output directories
    output_dir = os.path.dirname(args.file)
    train_dir = os.path.join(output_dir, 'train')


    # Copy the input CSV file to train and test directories
    train_csv = os.path.join(output_dir, args.trainname)
    test_csv = os.path.join(output_dir, args.testname)
    shutil.copyfile(args.file, train_csv)
    shutil.copyfile(args.file, test_csv)

    # Calculate the number of testing rows
    total_rows = sum(1 for line in open(args.file)) - 1  # Subtract 1 for header row
    testing_rows = int(total_rows * args.testpercent)

    # Shuffle and split the rows
    with open(args.file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        data = list(reader)
        random.shuffle(data)

        # Write testing rows to testing.csv
        with open(test_csv, 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerow(header)
            writer.writerows(data[:testing_rows])

        # Write remaining rows to training.csv
        with open(train_csv, 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(header)
            writer.writerows(data[testing_rows:])

    return train_csv, test_csv


if __name__ == '__main__':
    main()

# # Example usage
# csv_file = './mutation/BTF_all.csv'
# train_csv, test_csv = split_csv(csv_file, 0.2)  # 20% of data for testing
#
# print(f"Training CSV: {train_csv}")
# print(f"Testing CSV: {test_csv}")


>>>>>>> 959c822f3e294e28bffabe791f7b4ec2e6720746
