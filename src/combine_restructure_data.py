# Importing necessary libraries
import jsonlines
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--input', help='Input file path')

args = argparser.parse_args()

# Number of files to combine
print("Combining {} files".format(len(args.input.split(','))))

# Reading the input file
files = args.input.split(',')
for i in range(len(files)):
    print("Reading file {}".format(files[i]))
    with jsonlines.open(files[i]) as f:
        if i == 0:
            output = list(f)
        else:
            output.extend(list(f))

# Writing the output file
print("Writing the output file")
with jsonlines.open('./data/combined_data.json', mode='w') as writer:
    writer.write_all(output)
