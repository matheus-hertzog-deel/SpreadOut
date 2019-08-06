import csv
import torch


class CSVLogger():
    def __init__(self, fieldnames, path):
        self.path = path
        self.csv_file = open(path, 'w+')

        # Write model configuration at top of csv
        #writer = csv.writer(self.csv_file)
        #writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


    def log_distances(self, stage, layers, check_fn=None):
        """
        Logs accumulated distance in layer (Euclidean)
        """
        if check_fn is None:
            check_fn = check_euclidean_sum
        with torch.no_grad():
            for l,layer in enumerate(layers):
                sum_dist = check_fn(layer)
                self.writerow({'Layer':l, 'Distance':sum_dist, 'Stage': stage})
                print(":: Layer {} got distance of {}".format(l, sum_dist))
