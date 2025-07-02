import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
from sklearn.model_selection import train_test_split

	 
csv_header_options = [
    'no_headers',  # don't use CSV file column headers at all
    'headers_as_prompt_prefix',  # list column headers once as part of the prompt prefix
    'headers_as_item_prefix',  # preface each data value with the corresponding colum header
]


class CsvParser:
    def __init__(
        self,
        csv_path,  # path to input csv file
        y_column_names,  # list of y or target column names
        y_column_types,  # list of y column types: 'categorical' or 'numerical'
        header_option='headers_as_item_prefix',  # see csv_headers_options comments
        num_decimal_places=2,  # number of decimal places to round the floating point numbers in the csv file to
        remove_rows_with_missing_values=True,  # if True, remove any row with missing or Nan  values
        columns_to_ignore=[],  # list of columns to ignore/delete in the csv file
        shuffle = True,  # Whether or not to shuffle the data before splitting
        seed=0,  # seed used to split the train/test data
        break_str='\n',
        column_separator=';',
        name_value_separator=':',
        missing_fraction=0.0,
    ) -> None:
        np.random.seed(seed)
        self.df = pd.read_csv(csv_path)
        self.header_options = header_option
        self.y_column_names = y_column_names
        self.y_column_types = y_column_types
        self.data = None
        self.seed = seed
        self.shuffle = shuffle
        self.break_str = break_str
        self.column_separator = column_separator
        self.name_value_separator = name_value_separator
        self.missing_fraction = missing_fraction
        self.mask = None
        self.num_decimal_places = num_decimal_places

        # delete any ignored columns
        if len(columns_to_ignore) > 0:
            self.df = self.df.drop(columns=columns_to_ignore)   

        # get the names of the x columns
        self.all_column_names = list(self.df.columns.values)  # first get the names of all the columns
        self.x_column_names = self.all_column_names.copy()  # make a copy of the all list
        for name in self.y_column_names:  # then remove the y columns
            self.x_column_names.remove(name)

        # remove rows with missing values
        if remove_rows_with_missing_values:
            self.df = self.df.dropna().reset_index(drop=True)

        # remove duplicate rows
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        # pandas maps int csv columns to float, so fix that
        for name in self.all_column_names:
            if is_float_dtype(self.df[name]):
                if all(x.is_integer() for x in self.df[name]):
                    self.df[name] = self.df[name].apply(np.int64)

        # round columns with floats to specified number of decimal places
        for name in self.all_column_names:
            if is_float_dtype(self.df[name]):
                self.df[name] = self.df[name].apply(lambda x: round(x, num_decimal_places) if pd.notna(x) else x)


        if self.missing_fraction > 0.0:  # generate 2D mask with missing_fraction probability
            self.mask = np.random.choice([0, 1], size=(len(self.df.index), len(self.x_column_names)), p=[1 - self.missing_fraction, self.missing_fraction])
            completely_missing_rows = []
            for row in range(len(self.df.index)):
                if len(np.where(self.mask[row] == 1)[0]) == len(self.x_column_names):  # all features would be missing
                    completely_missing_rows.append(row)
            self.mask = np.ascontiguousarray(np.delete(self.mask, completely_missing_rows, axis=0))
            self.df = self.df.drop(completely_missing_rows, axis=0)
            self.df = self.df.reset_index()

    def get_data_test_fraction(
        self,
        test_fraction=0.2,
        train_size_limit=None,
        test_size_limit=None
    ):
        xs = []
        ys = []
        for i, _ in self.df.iterrows():
            x = ''
            for j, name in enumerate(self.x_column_names):
                if (self.missing_fraction > 0.0) and (self.mask[i, j] == 1):
                        continue
                if self.header_options == 'headers_as_item_prefix':
                    x += ('{}{}{}{}'.format(name, self.name_value_separator, self.df[name][i], self.column_separator))
                else:
                    x += ('{}{}'.format(self.df[name][i], self.column_separator))
                    
            y = []
            for name in self.y_column_names:
                y.append(self.df[name][i])
            xs.append(x)
            ys.append(y)

        x_all = np.array(xs)
        y_all = np.array(ys, dtype=object)

        # train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_all,
            y_all,
            test_size=test_fraction,
            random_state=self.seed,
            shuffle=self.shuffle
        )
        
        self.data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'x_true': x_all,
            'y_true': y_all,
        }

        return self._truncate(train_size_limit=train_size_limit, test_size_limit=test_size_limit)
    
    def get_data_shots_per_class(
        self,
        shots=1,
        train_size_limit=None,
        test_size_limit=None
    ):
        # check that y is categorical and there is only one y column
        assert len(self.y_column_names) == 1
        assert self.y_column_types[0] == 'categorical'
        xs = []
        ys = []
        for i, _ in self.df.iterrows():
            x = ''
            for j, name in enumerate(self.x_column_names):
                if (self.missing_fraction > 0.0) and (self.mask[i, j] == 1):
                        continue
                if self.header_options == 'headers_as_item_prefix':
                    x += ('{}{}{}{}'.format(name, self.name_value_separator, self.df[name][i], self.column_separator))
                else:
                    x += ('{}{}'.format(self.df[name][i], self.column_separator))
                    
            y = []
            for name in self.y_column_names:
                y.append(self.df[name][i])
            xs.append(x)
            ys.append(y)

        x_all = np.array(xs)
        y_all = np.array(ys, dtype=object)

        # train/test split
        labels = self.df[self.y_column_names[0]]
        x_train, x_test, y_train, y_test = train_test_split(
            x_all,
            y_all,
            train_size=shots * len(np.unique(labels)),
            test_size=None,
            random_state=self.seed,
            shuffle=self.shuffle,
            stratify=labels
        )
        
        self.data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'x_true': x_all,
            'y_true': y_all,
        }

        return self._truncate(train_size_limit=train_size_limit, test_size_limit= test_size_limit)

    def get_data_fixed_indices(
        self,
        train_start_index,
        train_end_index,
        test_start_index,
        test_end_index,
        ):
        xs = []
        ys = []
        for i, _ in self.df.iterrows():
            x = ''
            for j, name in enumerate(self.x_column_names):
                if (self.missing_fraction > 0.0) and (self.mask[i, j] == 1):
                        continue
                if self.header_options == 'headers_as_item_prefix':
                    x += ('{}{}{}{}'.format(name, self.name_value_separator, self.df[name][i], self.column_separator))
                else:
                    x += ('{}{}'.format(self.df[name][i], self.column_separator))
                    
            y = []
            for name in self.y_column_names:
                y.append(self.df[name][i])
            xs.append(x)
            ys.append(y)

        x_all = np.array(xs)
        y_all = np.array(ys, dtype=object)      

        x_test = x_all[test_start_index:test_end_index]
        y_test = y_all[test_start_index:test_end_index, :]
        x_train = x_all[train_start_index:train_end_index]
        y_train = y_all[train_start_index:train_end_index, :]

        # shuffle the data and return
        train_permutation = np.random.permutation(len(x_train))
        test_permutation = np.random.permutation(len(x_test))

        self.data = {
            'x_train': x_train[train_permutation],
            'y_train': y_train[train_permutation],
            'x_test': x_test[test_permutation],
            'y_test': y_test[test_permutation],
        }

        return self.data

    def get_data_fixed_train_and_test_size(
        self,
        train_size_limit=None,
        test_size_limit=None,
    ):
        xs = []
        ys = []
        for i, _ in self.df.iterrows():
            x = ''
            for j, name in enumerate(self.x_column_names):
                if (self.missing_fraction > 0.0) and (self.mask[i, j] == 1):
                        continue
                if self.header_options == 'headers_as_item_prefix':
                    x += ('{}{}{}{}'.format(name, self.name_value_separator, self.df[name][i], self.column_separator))
                else:
                    x += ('{}{}'.format(self.df[name][i], self.column_separator))
                    
            y = []
            for name in self.y_column_names:
                y.append(self.df[name][i])
            xs.append(x)
            ys.append(y)

        x_all = np.array(xs)
        y_all = np.array(ys, dtype=object)

        x_train_start = []  
        y_train_start = []
        column_index = -1  # if the last column is categorical, we will ensure that there is at least one example per class in the training set
        classes = []
        if self.y_column_types[column_index] == 'categorical':
            # ensure that there is a least 1 example per class in the train set
            classes = np.unique(y_all[:, column_index])
            for cls in classes:
                class_indices = np.where(y_all[:, column_index] == cls)[0]
                class_index = np.random.choice(class_indices, 1, replace=False)
                x_train_start.append(x_all[class_index])
                x_all = np.delete(x_all, class_index, axis=0)
                y_train_start.append(y_all[class_index, :])
                y_all = np.delete(y_all, class_index, axis=0)
            x_train_start = np.array(x_train_start).squeeze(axis=1)
            y_train_start = np.array(y_train_start).squeeze(axis=1)

        # train/test split
        if (train_size_limit - len(x_train_start)) > len(classes):
            x_train, x_test, y_train, y_test = train_test_split(
                x_all,
                y_all,
                train_size=train_size_limit - len(x_train_start),
                test_size=test_size_limit,
                random_state=self.seed,
                shuffle=self.shuffle,
                stratify=y_all[:, column_index] if self.y_column_types[column_index] == 'categorical' else None
            )
            if len(x_train_start) > 0:
                x_train = np.concatenate((x_train_start, x_train))
                y_train = np.concatenate((y_train_start, y_train))
        else:
            # draw the remaining training items at random from x_all, y_all
            permutation = np.random.permutation(len(y_all))
            train_samples_to_draw = train_size_limit - len(x_train_start)
            if train_samples_to_draw > 0:
                x_train = np.concatenate((x_train_start, x_all[permutation][0 : train_samples_to_draw]))
                y_train = np.concatenate((y_train_start, y_all[permutation][0 : train_samples_to_draw]))
            else:
                x_train = x_train_start
                y_train = y_train_start
            # draw the test examples at random from the remaining x_all, y_all
            x_test = x_all[permutation][train_samples_to_draw : test_size_limit + train_samples_to_draw]
            y_test = y_all[permutation][train_samples_to_draw : test_size_limit + train_samples_to_draw]
        
        self.data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        }

        return self.data
