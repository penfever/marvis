import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np


class ImputationCsvParser:
    def __init__(self,
                 csv_path,  # path to input csv file
                 num_decimal_places=2,  # number of decimal places to round the floating point numbers in the csv file to
                 columns_to_ignore=[],  # list of columns to ignore/delete in the csv file
                 column_separator=';',
                 name_value_separator=':',
                 header_option='headers_as_item_prefix',
                 ) -> None:
        self.df = pd.read_csv(csv_path)
        self.column_separator = column_separator
        self.name_value_separator = name_value_separator
        self.header_option = header_option
        self.rearranged_column_names = []

        # delete any ignored columns
        if len(columns_to_ignore) > 0:
            self.df = self.df.drop(columns=columns_to_ignore)   

        # pandas maps int csv columns to float, so fix that
        for column in self.df:
            if is_float_dtype(self.df[column]):
                if all(x.is_integer() for x in self.df[column]):
                    self.df[column] = self.df[column].apply(np.int64)

        # round columns with floats to specified number of decimal places
        for column in self.df:
            if is_float_dtype(self.df[column]):
                self.df[column] = self.df[column].apply(lambda x: round(x, num_decimal_places) if pd.notna(x) else x)

        self.all_column_names = list(self.df.columns.values)  # first get the names of all the columns

    def get_all_data(self):
        return self.df.to_numpy(dtype=object, copy=True)
    
    def get_column_names(self):
        return self.all_column_names.copy()
    
    def get_data(
        self,
        y_column_names=[],
        missing_mask=None,
        test_row=None            
    ):
        # mutate the colums such that the known columns are at the front
        missing_column_indices = np.where(missing_mask[test_row] == True)[0]
        missing_column_names = []
        for index in sorted(missing_column_indices):
            missing_column_names.append(self.all_column_names[index])

        present_column_names = self.all_column_names.copy()
        for index in sorted(missing_column_indices, reverse=True):
            del present_column_names[index]

        if len(present_column_names) == 0:
            return None
        
        self.rearranged_column_names = present_column_names + missing_column_names

        xs = []
        ys = []
        for i, _ in self.df.iterrows():
            x = ''
            for j, name in enumerate(self.rearranged_column_names):
                if missing_mask[i, self.all_column_names.index(name)]:  # need to swizzle the mask indices as well
                        continue
                if self.header_option == 'headers_as_item_prefix':
                    x += ('{}{}{}{}'.format(name, self.name_value_separator, self.df[name][i], self.column_separator))
                else:
                    x += ('{}{}'.format(self.df[name][i], self.column_separator))
                    
            # since there is no y per se, and we won't be adding y, remove the last column separator
            x = x[:-1]

            y = []
            for name in y_column_names:
                y.append(self.df[name][i])
            xs.append(x)
            ys.append(y)

        x_all = np.array(xs)
        y_all = np.array(ys)

        data = {
            'x_train': np.delete(x_all, test_row, axis=0),
            'y_train': np.delete(y_all, test_row, axis=0),
            'x_test': np.expand_dims(x_all[test_row] + self.column_separator, axis=0),  # do put a column separator on x_test
            'y_test': np.expand_dims(y_all[test_row], axis=0),
        }

        return data
    
    def put_data(self, row, column_name, value):
        self.df.at[row, column_name] = value
        assert self.df.at[row, column_name] == value

    def get_rearranged_column_names(self):
        return self.rearranged_column_names.copy()
