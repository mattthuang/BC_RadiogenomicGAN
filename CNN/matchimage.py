<<<<<<< HEAD
import os
import csv
import shutil
import argparse

def matchimage(args):
    # Define paths to CSV file and image directory
    csv_file = args.csvfile
    img_dir = args.imagedir

    # Define paths to output directory
    output_dir = args.outputdir
    # output_dir = "./training_image2/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open CSV file and read rows
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            # Get image filename from first column of CSV
            img_filename = row[0] + ".npy"
            print(img_filename)
            # Construct full path to image file
            img_path = os.path.join(img_dir, img_filename)

            # Check if image file exists
            if os.path.isfile(img_path):
                # Construct full path to output file
                output_path = os.path.join(output_dir, img_filename)

                # Copy image file to output directory
                shutil.copy2(img_path, output_path)
            else:
                print("no files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvfile', type=str, help='BTF features')
    parser.add_argument('--imagedir', type=str, help='MRI directory')
    parser.add_argument('--outputdir', type = str, help = 'output directory')
    args = parser.parse_args()
    matchimage(args)

if __name__ == '__main__':
=======
import os
import csv
import shutil
import argparse

def matchimage(args):
    # Define paths to CSV file and image directory
    csv_file = args.csvfile
    img_dir = args.imagedir

    # Define paths to output directory
    output_dir = args.outputdir
    # output_dir = "./training_image2/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open CSV file and read rows
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            # Get image filename from first column of CSV
            img_filename = row[0] + ".npy"
            print(img_filename)
            # Construct full path to image file
            img_path = os.path.join(img_dir, img_filename)

            # Check if image file exists
            if os.path.isfile(img_path):
                # Construct full path to output file
                output_path = os.path.join(output_dir, img_filename)

                # Copy image file to output directory
                shutil.copy2(img_path, output_path)
            else:
                print("no files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvfile', type=str, help='BTF features')
    parser.add_argument('--imagedir', type=str, help='MRI directory')
    parser.add_argument('--outputdir', type = str, help = 'output directory')
    args = parser.parse_args()
    matchimage(args)

if __name__ == '__main__':
>>>>>>> 959c822f3e294e28bffabe791f7b4ec2e6720746
    main()