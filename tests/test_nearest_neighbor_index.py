"""nn_search_test"""

import random
import time
import unittest

from pynn import NearestNeighborIndex

# Test Refactoring and Improvements:
#
# The tests have been significantly refactored to cover a wider range of cases and include benchmarks
# across varying dataset sizes (up to 10,000 index points). Note that the dataset larger than 50,000 points
# does not finish on my machine. The refactored tests are now broken into subtests, providing detailed feedback
# on specific failures, making debugging easier and more efficient.
#
# Key Improvements:
# 1. **Expanded Functional Coverage**:
#    - The `test_basic` function now tests many more edge cases, ensuring robust correctness.
#    - Subtests have been added to isolate failures for better diagnostic clarity.
#
# 2. **Enhanced Benchmarking**:
#    - The `test_benchmark` function evaluates performance across a variety of dataset sizes,
#      including small, medium, and large datasets, to measure scalability.
#
# 3. **Improved Test Structure**:
#    - Subtests separate functionality testing (`test_basic`) from performance testing (`test_benchmark`),
#      enabling a clearer focus for each type of validation.
#
# Future Improvements:
# - **Separate Tests for Fast and Slow Methods**:
#   Refactor `test_basic` into two separate tests (`test_basic_slow` and `test_basic_fast`)
#   to directly compare results from the two methods.
#
# - **Time Limits for Large Datasets**:
#   Introduce time limits for large datasets to skip tests that are impractically long, while still
#   noting that such cases need optimization.
#
# - **More Edge Cases**:
#   Add performance-specific edge cases, such as clustered points, sparse points, identical points,
#   and empty datasets, to stress-test the implementation further.
#
# While there is still room for additional improvements, these updates represent a substantial enhancement
# to the existing test suite, offering broader coverage and clearer diagnostics while maintaining performance goals.


class NearestNeighborIndexTest(unittest.TestCase):
    def test_basic(self):
        """
        test_basic tests a handful of nearest neighbor queries, iterating through test cases
        and running multiple subtests for each case using the slow method.
        """

        # Define the test points
        test_points = [
            (1, 2),
            (1, 0),
            (10, 5),
            (-1000, 20),
            (3.14159, 42),
            (42, 3.14159),
            (0, 0),
            (1000, -1000),
            (-500, -500),
            (1.5, 1.5),
            (2, 2),
        ]

        # Define test cases with query points and their expected results
        test_cases = [
            # Standard cases
            {"query_point": (0, 0), "expected_result": (0, 0)},  # Nearest is at origin
            {"query_point": (1, 1), "expected_result": (1.5, 1.5)},  # Nearest is above query
            {"query_point": (-2000, 0), "expected_result": (-1000, 20)},  # Far negative
            {"query_point": (40, 3), "expected_result": (42, 3.14159)},  # Close to large value
            {"query_point": (2, 2), "expected_result": (2, 2)},  # Query is exactly a test point

            # Edge and boundary cases
            {"query_point": (1000, -1000), "expected_result": (1000, -1000)},  # Query matches a far test point
            {"query_point": (-500, -499), "expected_result": (-500, -500)},  # Close to large negative
            {"query_point": (1.6, 1.6), "expected_result": (1.5, 1.5)},  # Near to (1.5, 1.5)
            {"query_point": (5, 5), "expected_result": (2, 2)},  # Closest on the same quadrant
            {"query_point": (-1, -1), "expected_result": (0, 0)},  # Nearest is origin

            # Randomly distributed test queries
            {"query_point": (15, 15), "expected_result": (10, 5)},
            {"query_point": (1, 3), "expected_result": (1, 2)},
            {"query_point": (0.1, 0.1), "expected_result": (0, 0)},
        ]
        # Outer loop: Iterate through test cases
        for case in test_cases:
            query_point = case["query_point"]
            expected_result = case["expected_result"]

            with self.subTest(subtest="Check exact match", query_point=query_point):
                uut = NearestNeighborIndex(query_point, test_points)
                result = uut.find_nearest_slow()
                result_tuple = tuple(result)
                self.assertEqual(result_tuple, expected_result)

            with self.subTest(subtest="Check result is valid point", query_point=query_point):
                uut = NearestNeighborIndex(query_point, test_points)
                result = uut.find_nearest_slow()
                result_tuple = tuple(result)
                self.assertEqual(result_tuple, expected_result)

            with self.subTest(subtest="Check result type", query_point=query_point):
                uut = NearestNeighborIndex(query_point, test_points)
                result = uut.find_nearest_slow()
                result_tuple = tuple(result)
                self.assertEqual(result_tuple, expected_result)

    def test_benchmark(self):
        """
        test_benchmark tests the slow and fast versions of the nearest neighbor index
        across multiple dataset sizes to determine the effective speedup. It is split into
        subtests for different dataset sizes.
        """

        def rand_point():
            return (random.uniform(-1000, 1000), random.uniform(-1000, 1000))

        # Define dataset sizes for indexing points and query points
        test_sizes = [
            {"index_size": 100, "query_size": 10},     # Small dataset
            {"index_size": 1000, "query_size": 100},   # Medium dataset
            {"index_size": 10000, "query_size": 1000},  # Large dataset
            {"index_size": 50000, "query_size": 5000},  # Extra large dataset
        ]

        for size in test_sizes:
            index_size = size["index_size"]
            query_size = size["query_size"]

            # Generate random points
            index_points = [rand_point() for _ in range(index_size)]  # Points to index
            query_points = [rand_point() for _ in range(query_size)]  # Query points

            print(f"\nTesting with index size: {index_size}, query size: {query_size}")

            # Subtest 1: Measure the execution time for the slow method
            with self.subTest(method="slow", index_size=index_size, query_size=query_size):
                expected = []  # Results for the slow method
                start = time.time()
                for query_point in query_points:
                    uut = NearestNeighborIndex(query_point, index_points)  # Create new UUT
                    expected.append(uut.find_nearest_slow())
                slow_time = time.time() - start
                print(f"slow time: {slow_time:0.2f} sec")

            # Subtest 2: Measure the execution time for the fast method
            with self.subTest(method="fast", index_size=index_size, query_size=query_size):
                actual = []  # Results for the fast method
                start = time.time()
                for query_point in query_points:
                    uut = NearestNeighborIndex(query_point, index_points)  # Create new UUT
                    actual.append(uut.find_nearest_fast())
                fast_time = time.time() - start
                print(f"fast time: {fast_time:0.2f} sec")

            # Calculate and print speedup
            speedup = slow_time / fast_time
            print(f"speedup: {speedup:0.2f}x")
