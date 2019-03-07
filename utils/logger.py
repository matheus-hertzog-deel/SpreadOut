import csv
import torch


class CSVLogger():
    def __init__(self, fieldnames, path):
        self.path = path
        self.csv_file = open(path, 'w+')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


    def log_distances(self, stage, layers):
        """
        Logs accumulated distance in layer (Euclidean)
        """
        for l,layer in enumerate(layers):
            sum_dist = 0
            for i,target in enumerate(layer.weight):
                for it, _filter in enumerate(layer.weight):
                    if i == it:
                        continue
                    dist = 0
                    sub = _filter - target
                    dist = torch.pow(sub,2)
                    dist = torch.sum(dist)
                    dist = torch.sqrt(dist)
                    sum_dist += dist
                    self.writerow({'Layer':l, 'Distance':dist.item(), 'Stage': stage})
            print(":: Layer {} got distance of {}".format(l, sum_dist))
