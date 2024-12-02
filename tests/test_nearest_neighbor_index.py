"""nn_search_test"""

import random
import time
import unittest

from pynn import NearestNeighborIndex


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
        ]

        # Define test cases with query points and their expected results
        test_cases = [
            {"query_point": (0, 0), "expected_result": (1, 0)},
            {"query_point": (-2000, 0), "expected_result": (-1000, 20)},
            {"query_point": (40, 3), "expected_result": (42, 3.14159)},
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
        test_benchmark tests a bunch of values using the slow and fast version of the index
        to determine the effective speedup, split into subtests for slow and fast methods.
        """

        def rand_point():
            return (random.uniform(-1000, 1000), random.uniform(-1000, 1000))

        index_points = [rand_point() for _ in range(10000)]  # Points to index
        query_points = [rand_point() for _ in range(1000)]   # Query points

        # Subtest 1: Measure the execution time for the slow method
        with self.subTest(method="slow"):
            expected = []  # Results for the slow method
            start = time.time()
            for query_point in query_points:
                uut = NearestNeighborIndex(query_point, index_points)  # Create new UUT
                expected.append(uut.find_nearest_slow())
            slow_time = time.time() - start
            print(f"slow time: {slow_time:0.2f} sec")

        # Subtest 2: Measure the execution time for the fast method
        with self.subTest(method="fast"):
            actual = []  # Results for the fast method
            start = time.time()
            for query_point in query_points:
                uut = NearestNeighborIndex(query_point, index_points)  # Create new UUT
                actual.append(uut.find_nearest_fast())
            new_time = time.time() - start
            print(f"new time: {new_time:0.2f} sec")

        # Calculate and print speedup
        speedup = slow_time / new_time
        print(f"speedup: {speedup:0.2f}x")
