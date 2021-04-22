import mysklearn.myutils as myutils

"""
Programmer: Alex Giacobbi
Class: CPSC 322-02, Spring 2021
Programming Assignment #2
02/16/21
Description: This program models 2D table with basic data preparation functionality.
MyPyTable has two attributes that represent a table header and the corresponding 
data. 
"""


import copy
import csv 
import statistics
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)


    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)
        

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """

        column_index = self.column_names.index(col_identifier)
        column_values = []

        for row in self.data:
            if row[column_index] != "NA" or include_missing_values:
                column_values.append(row[column_index])

        return column_values


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                except ValueError as v:
                    pass


    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row in rows_to_drop:
            try:
                self.data.remove(row)
            except ValueError as v:
                pass


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        self.data = []

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.data.append(row)

        self.column_names = self.data.pop(0)
        self.convert_to_numeric()
        
        return self 


    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)


    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """

        key_column_index = [self.column_names.index(key_name) for key_name in key_column_names]
        unique_keys = set()
        duplicates = []

        for row in self.data:
            key = []
            for index in key_column_index:
                key.append(row[index])
            if tuple(key) in unique_keys:
                duplicates.append(row)
            else:
                unique_keys.add(tuple(key))

        return duplicates


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        rows_to_drop = []
        
        for row in self.data:
            for value in row:
                if value == "NA":
                    rows_to_drop.append(row)

        self.drop_rows(rows_to_drop)


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        column_values = self.get_column(col_name, include_missing_values=False)
        average_value = sum(column_values) / len(column_values)
        
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = average_value


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable. The stats included
        are min, max, mid, avg, median.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed.
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats = []

        for column in col_names:
            column_stats = []
            column_data = self.get_column(column, include_missing_values=False)

            if len(column_data) != 0:
                column_stats.append(column)
                column_stats.append(min(column_data))
                column_stats.append(max(column_data))
                column_stats.append((min(column_data) + max(column_data)) / 2)
                column_stats.append(sum(column_data) / len(column_data))
                column_stats.append(statistics.median(column_data))
                stats.append(column_stats)

        return MyPyTable(column_names=header, data=stats)


    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        merged_data = []
        new_header = copy.deepcopy(self.column_names)
        header_rest = copy.deepcopy(other_table.column_names)
        header_rest = [col_name for col_name in header_rest if not col_name in key_column_names]
        new_header = new_header + header_rest

        for row_l in self.data:
            for row_r in other_table.data:
                # Look for matching keys
                found_match = True
                rest_of_row = copy.deepcopy(row_r)
                for i in range(len(key_column_names)):
                    left_table_column_index = self.column_names.index(key_column_names[i])
                    right_table_column_index = other_table.column_names.index(key_column_names[i])
                    if row_l[left_table_column_index] != row_r[right_table_column_index]:
                        found_match = False
                    rest_of_row.pop(right_table_column_index - i)

                # Add row
                if found_match:
                    merge_row = [value for value in row_l]
                    for value in rest_of_row:
                        merge_row.append(value)
                    merged_data.append(merge_row)

        return MyPyTable(column_names=new_header, data=merged_data)


    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        merged_data = []
        new_header = copy.deepcopy(self.column_names)
        header_rest = copy.deepcopy(other_table.column_names)
        header_rest = [col_name for col_name in header_rest if not col_name in key_column_names]
        new_header = new_header + header_rest

        for row_l in self.data:
            match_exists = False
            for row_r in other_table.data:
                # Look for matching keys
                possible_match = True
                rest_of_row = copy.deepcopy(row_r)
                for i in range(len(key_column_names)):
                    left_table_column_index = self.column_names.index(key_column_names[i])
                    right_table_column_index = other_table.column_names.index(key_column_names[i])
                    if row_l[left_table_column_index] != row_r[right_table_column_index]:
                        possible_match = False
                    rest_of_row.pop(right_table_column_index - i)

                # Add row
                if possible_match:
                    match_exists = True
                    merge_row = [value for value in row_l]
                    for value in rest_of_row:
                        merge_row.append(value)
                    merged_data.append(merge_row)

            if not match_exists:
                merge_row = [value for value in row_l]
                for column in header_rest:
                    merge_row.append("NA")
                merged_data.append(merge_row)

        for row_r in other_table.data:
            match_exists = False
            for existing_row in merged_data:
                # Look for matching keys
                possible_match = True
                rest_of_row = copy.deepcopy(row_r)
                for i in range(len(key_column_names)):
                    left_table_column_index = new_header.index(key_column_names[i])
                    right_table_column_index = other_table.column_names.index(key_column_names[i])
                    if existing_row[left_table_column_index] != row_r[right_table_column_index]:
                        possible_match = False
                    rest_of_row.pop(right_table_column_index - i)

                if possible_match:
                    match_exists = True

            if not match_exists:
                # fill N/A
                merge_row = ["NA" for value in new_header]
                # replace with values
                for col_name in other_table.column_names:
                    new_table_index = new_header.index(col_name)
                    right_table_index = other_table.column_names.index(col_name)
                    merge_row[new_table_index] = row_r[right_table_index]

                merged_data.append(merge_row)

        return MyPyTable(column_names=new_header, data=merged_data)