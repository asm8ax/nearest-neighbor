import math
import numba
import numpy as np
from scipy.spatial import KDTree


class NearestNeighborIndex:
    """
    NearestNeighborIndex is a class for finding the nearest neighbor to a query point in a set of 2D points.

    This class provides methods for efficient nearest neighbor searches using a sphere-based approach with
    iterative radius reduction, and a fallback to a brute-force method for robustness. It is optimized for
    handling datasets of moderate size and ensures correctness even in edge cases.

    Parameters
    ----------
    query_point : tuple of float
        A tuple representing the query point (x, y).
    points : list of tuple of float
        A list of 2D points to be indexed, where each point is represented as a tuple (x, y).

    Attributes
    ----------
    points : np.ndarray
        A NumPy array containing the indexed 2D points.
    query_point : np.ndarray
        A NumPy array representing the query point.

    Methods
    -------
    find_nearest_fast()
        Uses a sphere-based search with radius halving and a greedy fallback to find the nearest neighbor.
    find_nearest_slow()
        Uses a brute-force approach to find the nearest neighbor by computing Euclidean distances to all points.
    compute_bounding_sphere()
        Calculates the initial bounding sphere for the dataset, centered at the query point.
    count_points_within_sphere(radius)
        Counts the number of points within a given radius using a KD-Tree.
    halve_radius_and_update(radius, point_counter)
        Reduces the radius of the bounding sphere and determines whether to continue or end the search.

    Notes
    -----
    - The class is designed to handle 2D datasets efficiently while maintaining correctness in nearest neighbor searches.
    - Advanced indexing structures like KD-Trees are used for efficient spatial querying, and a brute-force fallback ensures
      robustness for edge cases.
    """

    def __init__(self, query_point, points):
        """
        Initialize the NearestNeighborIndex with a query point and a list of 2D points.

        Parameters
        ----------
        query_point : tuple of float
            A tuple representing the coordinates of the query point (x, y).
        points : list of tuple of float
            A list of 2D points to be indexed, where each point is represented as a tuple of two floats (x, y).

        Attributes
        ----------
        self.points : np.ndarray
            A NumPy array containing the indexed 2D points.
        self.query_point : np.ndarray
            A NumPy array representing the query point.

        Notes
        -----
        - The points are converted to a NumPy array for efficient numerical operations.
        - The query point is also stored as a NumPy array for consistency with the dataset.
        """
        self.points = np.array(points, dtype=float)
        self.query_point = np.array(query_point, dtype=float)
        self.kd_tree = KDTree(self.points)

    def find_nearest_slow(self):
        """
        Find the nearest point to the query point using a brute-force search.

        This method computes the Euclidean distance between the query point and each 
        point in the dataset to find the nearest neighbor.

        Parameters
        ----------
        None
            This method uses `self.query_point` as the query point and `self.points` 
            as the dataset to search through.

        Returns
        -------
        tuple of float
            The coordinates of the nearest neighbor point as a tuple (x, y).

        Notes
        -----
        - The method calculates the Euclidean distance between the query point and each 
          point in the dataset, checking all points in a brute-force manner.
        - The time complexity is O(n), where n is the number of points in the dataset.
        - This implementation is straightforward but less efficient compared to sphere-based or KD-Tree searches.
        """

        min_dist = None
        min_point = None

        for point in self.points:
            deltax = point[0] - self.query_point[0]
            deltay = point[1] - self.query_point[1]
            dist = math.sqrt(deltax * deltax + deltay * deltay)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_point = point
#        print("The min_point is:")
#        print(min_point)
        return min_point

    def find_nearest_fast(self):
        """
        Find the nearest point to the query point using a sphere-based search and greedy fallback.

        This method iteratively reduces the search radius of a bounding sphere to narrow
        down the nearest neighbor. If necessary, it falls back to a brute-force search for robustness.

        Parameters
        ----------
        None
            This method uses `self.query_point` as the fixed query point and `self.points`
            as the set of points to search.

        Returns
        -------
        tuple of float
            The coordinates of the nearest neighbor point as a tuple (x, y).

        Notes
        -----
        - The sphere-based search progressively narrows down the number of points by halving the radius.
        - If only one point remains in the sphere, it is identified as the nearest neighbor.
        - In case no points remain within the sphere, the method reverts to a brute-force greedy search.
        """
        # Initialize bounding sphere
        _, radius, farthest_point = self.compute_bounding_sphere()

        # KD-Tree to efficiently query points within the sphere
        kd_tree = KDTree(self.points)

        while True:
            # Find points within the current radius using KD-Tree
            indices = kd_tree.query_ball_point(self.query_point, radius)
            point_count = len(indices)

            if point_count == 1:
                # Only one point remains, it's the nearest neighbor
                return tuple(self.points[indices[0]])
            elif point_count == 0 or radius < 1e-6:
                # No points within radius or radius is too small
                break

            # Halve the radius
            radius /= 2

        # Fallback: Brute-force search among all points
        closest_idx = kd_tree.query(self.query_point)[1]
        return tuple(self.points[closest_idx])

    def compute_bounding_sphere(self):
        """
        Wrapper for the compute_bounding_sphere function.

        This method computes the bounding sphere for the dataset, with the query point
        as the fixed center of the sphere. It returns the center, radius, and the farthest 
        point from the center.
        """
        return compute_bounding_sphere(self.query_point, self.points)

    def count_points_within_sphere(self, radius):
        """
        Count the number of points within a bounding sphere using a KD-Tree.

        This function uses a KD-Tree to efficiently determine how many points in the dataset 
        fall within a sphere of a given radius, centered at the query point.

        Parameters
        ----------
        radius : float
            The radius of the sphere within which points are counted.

        Returns
        -------
        point_counter : int
            The total number of points within the bounding sphere.
        has_more_than_one : bool
            A flag indicating whether more than one point lies within the sphere.

        Notes
        -----
        - The KD-Tree is built using the dataset of points provided during the initialization of the class.
        - If only one point remains within the sphere, it is considered the nearest neighbor.
        - This function is designed to work in 2D or higher-dimensional spaces.
        """
        indices = self.kd_tree.query_ball_point(self.query_point, radius)
        point_counter = len(indices)
        return point_counter, len(indices) > 1

    def halve_radius_and_update(self, radius, point_counter):
        """
        Wrapper for the halve_radius_and_update function.

        This method reduces the radius of the bounding sphere by half, checks the updated
        number of points within the sphere, and determines if the search should end or continue.
        """
        radius, ended = halve_radius_and_update(self.points, self.query_point, radius, point_counter)
        if ended:
            return None
        self.point_count = point_counter  # Update point count if needed
        return radius


@numba.jit(nopython=True)
def compute_bounding_sphere(query_point, points_array):
    """
    Compute a bounding sphere with a fixed center (query point) that encapsulates all points,
    using an approximate method to find the farthest point.

    Parameters
    ----------
    query_point : np.ndarray
        The fixed center of the sphere, represented as a NumPy array of shape (2,).
    points_array : np.ndarray
        A NumPy array of shape (n, 2) containing the set of points to encapsulate, 
        where each row is a 2D point (x, y).

    Returns
    -------
    tuple
        center : np.ndarray
            The center of the sphere, which is the same as the query_point.
        radius : float
            The radius of the sphere, calculated as the distance from the center to the farthest point.
        farthest_point : np.ndarray
            The farthest point from the center within the given points_array.

    Notes
    -----
    - This function preallocates arrays for the extreme points (max_x, min_x, max_y, min_y) to minimize overhead.
    - It calculates the radius as the Euclidean distance between the query point and the farthest point.
    """
    # Preallocate arrays for extreme points
    max_x = points_array[0]
    min_x = points_array[0]
    max_y = points_array[0]
    min_y = points_array[0]

    for i in range(len(points_array)):
        if points_array[i, 0] > max_x[0]:
            max_x = points_array[i]
        if points_array[i, 0] < min_x[0]:
            min_x = points_array[i]
        if points_array[i, 1] > max_y[1]:
            max_y = points_array[i]
        if points_array[i, 1] < min_y[1]:
            min_y = points_array[i]

    # Create a fixed-size array for candidate points
    candidate_points = np.zeros((4, 2), dtype=np.float64)
    candidate_points[0] = max_x
    candidate_points[1] = min_x
    candidate_points[2] = max_y
    candidate_points[3] = min_y

    # Compute squared distances to the center
    distances_squared = np.zeros(4, dtype=np.float64)
    for i in range(4):
        dx = candidate_points[i, 0] - query_point[0]
        dy = candidate_points[i, 1] - query_point[1]
        distances_squared[i] = dx * dx + dy * dy

    # Find the farthest point and the radius
    farthest_index = np.argmax(distances_squared)
    farthest_point = candidate_points[farthest_index]
    radius = math.sqrt(distances_squared[farthest_index])

    return query_point, radius, farthest_point


@numba.jit(nopython=True)
def halve_radius_and_update(points, query_point, radius, point_counter):
    """
    Halve the sphere's radius, update the radius and point count, and handle special cases during nearest neighbor search.

    This function reduces the radius of the bounding sphere by half, determines the number of points within the new sphere,
    and checks for termination conditions. If no points remain or the search reaches a sufficiently small set of points,
    a brute-force fallback is triggered.

    Parameters
    ----------
    points : np.ndarray
        Array of 2D points where each row represents a point (x, y).
    query_point : np.ndarray
        The query point as a NumPy array (x, y).
    radius : float
        The current radius of the bounding sphere.
    point_counter : int
        The initial number of points within the bounding sphere.

    Returns
    -------
    radius : float or None
        The updated radius after halving, or None if the search has ended.
    ended : bool
        A flag indicating whether the search has ended.

    Notes
    -----
    - If only one point remains within the bounding sphere, the function outputs the nearest neighbor and ends the search.
    - If no points remain, the function reverts to a brute-force fallback to find the closest point.
    - If the remaining points are reduced to 5% or less of the initial count, the function switches to a greedy search.
    """

    previous_radius = radius
    radius /= 2
#    print("Sphere Halving Performed.")

    # Compute distances and count points within the new radius
    distances = np.sqrt(np.sum((points - query_point) ** 2, axis=1))
    current_point_counter = np.sum(distances <= radius)

    if current_point_counter == 1:
        nearest_neighbor = points[distances <= radius][0]
#        print("We have already reached our nearest neighbor!")
#        print(f"Nearest neighbor: {nearest_neighbor}")
        return None, True
    elif current_point_counter == 0:
        #        print("No points found in the current sphere. Reverting to the previous radius.")
        squared_distances = np.sum((points - query_point) ** 2, axis=1)
        closest_point_index = np.argmin(squared_distances)
#        print(f"Closest point (brute force): {points[closest_point_index]}")
        return None, True
    elif current_point_counter <= 0.05 * point_counter:
        #        print("5% of points remaining — switching to greedy algorithm.")
        squared_distances = np.sum((points - query_point) ** 2, axis=1)
        closest_point_index = np.argmin(squared_distances)
#        print(f"Closest point: {points[closest_point_index]}")
        return None, True

    return radius, False


# if __name__ == "__main__":
#    test_points = [
#        (1, 2),
#        (1, 0),
#        (10, 5),
#        (-1000, 20),
#        (3.14159, 42),
#        (42, 3.14159),
#    ]
#    query_point = (1, 0)
#    uut = NearestNeighborIndex(query_point, test_points)
#    uut.find_nearest_fast()
#    # uut.find_nearest_slow()


# def main():
#    main()
