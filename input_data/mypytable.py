##############################################
# Programmer: Joseph Torii and Alex Giaccobi
# Class: CptS 322-01, Spring 2021
# Semester Project
# 3/24/21
# 
# Description: This program computes a MyPyTable and has a bunch of functions
# that help create, update, and even save to a new file. These functions for 
# MyPyTable are then used in pa2.py.
##############################################

import copy
import csv 
import os
from statistics import median
from tabulate import tabulate

# from tabulate import tabulate # uncomment if you want to use the pretty_print() method
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
        """Prints the table in a nicely formatted grid structure"""

        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """

        numrows = len(self.data) # computes the rows
        numcols = len(self.data[0]) # computes the columns

        return numrows, numcols 


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

        # looping through the table
        for row in range(len(self.data)):
            for col in range(len(self.data[row])):
                # check if the value can be converted to a numeric type
                try:
                    self.data[row][col] = float(self.data[row][col])
                except:
                    pass
                    

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """

        for row in self.data:
            for drop in rows_to_drop:
                if row == drop:
                    self.data.remove(row)


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
        # opens the file
        with open(filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.column_names = next(csv_reader)

            # reading the csv file into self.data
            self.data = list(csv_reader)

        self.convert_to_numeric()

        return self


    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """

        with open(filename, 'w') as csv_file:
            csv_writter = csv.writer(csv_file)
            csv_writter.writerow(self.column_names)
            csv_writter.writerows(self.data)


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
        duplicate_list = []

        # get the indexes
        index_list = []
        for x in self.column_names:
            for y in key_column_names:
                if x == y:
                    index_list.append(self.column_names.index(x))

        seen_list = [] # a list to keep track of what we have seen so far in self.data

        for row in self.data:
            random_list = []
            for check in index_list:
                random_list.append(row[check])
            if random_list in seen_list:
                duplicate_list.append(row)
            else:
                seen_list.append(random_list)

        return duplicate_list


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """        
        for row in self.data:
            for col in row:
                if col == "NA":
                    self.data.remove(row)


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # get the index
        column_index = self.column_names.index(col_name)
        
        average = 0
        row_counter = 0
        # gets the column associated with col_name
        for row in self.data:
            if row[column_index] != 'NA':
                average = average + row[column_index] 
                row_counter += 1
       
        average = average / row_counter # getting the average

        # checking where there are empty rows and inserting the average
        for row in self.data:
            if row[column_index] == 'NA':
                row[column_index] = average                
        

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order 
            is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """        
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats = []

        for column in col_names:
            column_stats = []
            column_data = self.get_column(column, False)

            if len(column_data) != 0:
                column_stats.append(column)
                column_stats.append(min(column_data))
                column_stats.append(max(column_data))
                column_stats.append((min(column_data) + max(column_data)) / 2)
                column_stats.append(sum(column_data) / len(column_data))
                column_stats.append(median(column_data))
                stats.append(column_stats)

        return MyPyTable(header, stats)
    

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        # setting the new table header
        new_table_columns = [] 
        # checks the columns in self.data and other_table.data
        for x in self.column_names:
            new_table_columns.append(x)
        for y in other_table.column_names:
            if y not in new_table_columns:
                new_table_columns.append(y)

        # setting the rest of the data to the new table
        new_data = []
        for row1 in self.data:
            for row2 in other_table.data:
                match = True
                rest_of_row = copy.deepcopy(row2)

                # checking for matching keys
                for x in range(len(key_column_names)):
                    lcolumn_index = self.column_names.index(key_column_names[x]) # setting the left table column index 
                    rcolumn_index = other_table.column_names.index(key_column_names[x]) # setting the right table column index

                    if row1[lcolumn_index] != row2[rcolumn_index]:
                        match = False

                    rest_of_row.pop(rcolumn_index - x)
                
                # adding the row to the new table
                if match:
                    set_row = [val for val in row1]

                    for val in rest_of_row:
                        set_row.append(val)

                    new_data.append(set_row)
        
        return MyPyTable(new_table_columns, new_data)
                    

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
        # getting the data combined in outer join
        new_column_names = []
        new_data = []

        data = copy.deepcopy(other_table.column_names)
        self_atts = copy.deepcopy(self.column_names)
        self_data = copy.deepcopy(self.data)

        # data now only contains the combined data from self.data and other_table.data
        for row in key_column_names:
            if row in data and row in self_atts:
                data.remove(row)
                self_atts.remove(row)
        
        new_column_names = self.column_names + data 

        index_array = []
        # setting the rest of the data to the new table
        for row1 in self_data:
            self_keys = []
            match = False

            # Loop through names to get key values
            for name in key_column_names:
                index = self.column_names.index(name) 
                val = row1[index] 
                self_keys.append(val) 

            match_index = 0

            # Loop through other table
            for row2 in other_table.data:
                other_keys = [] # Initialize list to get key values

                # Loop through names to get key values
                for name in key_column_names:
                    index = other_table.column_names.index(name) 
                    val = row2[index] 
                    other_keys.append(val) 

                # Check if there is a match on the key values
                if self_keys == other_keys:
                    index_array.append(match_index)
                    match = True
                    other_copy = copy.deepcopy(row2)

                    # Get ride of duplicate key values
                    for val in other_keys:
                        other_copy.remove(val)

                    added_row = row1 + other_copy
                    new_data.append(added_row)
                    
                match_index += 1
                
            if (not match):
                # Need to add row from self to outer join (not in inner join)
                added_row = row1

                # adding the empty data
                for _ in data:
                    added_row.append("NA")
                new_data.append(added_row)

        empty_data = [] # List to hold empty data in other table
        kept_data = [] # List to hold the data in other table

        # setting the lists
        for x in range(len(new_column_names)):
            if not new_column_names[x] in other_table.column_names:
                # The column name is not in the other table, sets to empty
                empty_data.append(x)
            else:
                # The column name is in the other table, want to keep value
                kept_data.append(other_table.column_names.index(new_column_names[x]))

        other_data = copy.deepcopy(other_table.data)

        # delete extra data
        for row in other_data:
            for x in range(len(row)):
                if not x in kept_data:
                    del row[x]

        # make necessary data entries as empty
        for row in other_data:
            for x in empty_data:
                row.insert(x, "NA")

        # appending new rows to the data of the returned table
        for x in range(len(other_data)):
            if not x in index_array:
                new_data.append(other_data[x])

        return MyPyTable(new_column_names, new_data)