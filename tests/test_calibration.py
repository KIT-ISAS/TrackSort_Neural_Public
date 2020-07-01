import unittest

import tensorflow as tf
import numpy as np
import model
import data_manager

from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


config = {
    'separation_prediction': False,
    'custom_variance_prediction': False,

    'clear_state': True,
    'calibrate': True,
    'is_loaded': True,
    'model_path': '/home/daniel/Documents/19_20_WS/Praktikum_Anthro/Code/next_step_rnn/models/kendall2.h5',
    'distance_threshold': 0,
    'distance_confidence': 0.99,
    'batch_size': 64,
    'num_timesteps': 40,

    'verbose': 1,
    'mc_dropout': False,
    'mc_samples': 0,

    'kendall_loss': True,
    'diagrams_path': '.'
}


class MyTestCase(unittest.TestCase):
    num_time_steps = 40
    nan_value = 0
    batch_size = 64
    # ToDo: how to reference to downloadable assets?
    glob_pattern_dataset = '/home/daniel/Documents/19_20_WS/Praktikum_Anthro/Code/next_step_rnn/data/DEM_cylinder.csv'

    def setUp(self) -> None:
        self.csv_data_set = data_manager.CsvDataSet(self.glob_pattern_dataset,
                                                    timesteps=self.num_time_steps,
                                                    nan_value=self.nan_value,
                                                    batch_size=self.batch_size,
                                                    data_is_aligned=False,
                                                    rotate_columns=True,
                                                    normalization_constant=1)
        print('CSV dataset loaded')

        self.m = model.Model(config, self.csv_data_set)
        print('Model ready')

        self.calibration_data = self.m.calibration_data

    def test_calibration(self):
        expected_confs = np.arange(0, 1, 0.01)

        query_conf = 0.67
        corrected_conf = self.m._apply_isotonic_regression([query_conf])[0]
        print("query_conf=", query_conf, " corrected_conf=", corrected_conf)

        critical_value_old = chi2.ppf(query_conf, df=2)
        critical_value_new = chi2.ppf(corrected_conf, df=2)

        print("critical_value_old", critical_value_old)
        print("critical_value_new", critical_value_new)

        print('... Circa {}% der quadrierten Mahal. Distanzen soll kleiner sein als'.format(query_conf * 100),
              critical_value_new)
        N = self.calibration_data['squared_mahalanobis'].shape[0]
        ratio = np.count_nonzero(self.calibration_data['squared_mahalanobis'] <= critical_value_new) / N
        print("... tatsächlich sind es {}%.".format(ratio * 100))

        absolute_error = abs(query_conf - ratio)
        print("... Abweichung: {}%.".format(absolute_error * 100))

        self.assertLess(absolute_error, 0.02)

        r = critical_value_new / critical_value_old
        print('Variance recalibration factor r for confidence {}% is given by'.format(100 * query_conf), r)
        scaled_variances = r * self.calibration_data['prediction_variance']

        # Test
        scaled_squared_mhd = []
        for i in range(N):
            scaled_squared_mhd += [
                np.sum(
                    ((self.calibration_data['prediction'][i] - self.calibration_data['target'][i])**2) / scaled_variances[i]
                )
                ]
        scaled_squared_mhd = np.array(scaled_squared_mhd)
        ratio = np.count_nonzero(scaled_squared_mhd <= chi2.ppf(query_conf, df=2)) / N
        print("Mit den rekalibrierten Varianzen sind es {}%.".format(ratio * 100))

        self.assertLess(abs(ratio-query_conf), 0.02)


if __name__ == '__main__':
    unittest.main()
