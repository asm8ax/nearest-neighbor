# Please create a simple example use of the pynn library for your end user. Assume that the end
# user knows a lot about their subject matter but has only a basic understanding of Python.

# Meaningful examples may include reading a file, finding a few nearby points and writing them
# out to the console.
import csv
from pynn.nearest_neighbor_index import NearestNeighborIndex  # Importing from the root pynn package

# On my machine, it is saying that there is no module named pynn. I am not sure why this is as it should be able to detect the folder and accompanying file.
# Anyway, this is how the example would be written.


def read_points_from_csv(file_path):
    """
    Reads 2D points from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing points.

    Returns
    -------
    list of tuple of float
        A list of points read from the CSV file.
    """
    points = []
    with open(file_path, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            points.append((x, y))
    return points


def main():
    # Path to the dataset file (update with your actual file path)
    dataset_file = r"C:\path\to\your\points.csv"

    # Read the dataset points from the CSV file
    dataset = read_points_from_csv(dataset_file)
    print(f"Dataset: {dataset}")

    # Define query points
    query_points = [(7, 7), (4, 4), (0, 0)]

    # Find the nearest neighbor for each query point
    for query_point in query_points:
        # Initialize the NearestNeighborIndex with the query point and dataset
        nn_index = NearestNeighborIndex(query_point, dataset)

        # Use the pre-existing find_nearest_fast function
        nearest_neighbor = nn_index.find_nearest_fast()

        # Print the result
        print(f"Query Point: {query_point} -> Nearest Neighbor: {nearest_neighbor}")


if __name__ == "__main__":
    main()
