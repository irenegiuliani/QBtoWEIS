import unittest
from weis.dlc_driver.dlc_generator import DLCGenerator, get_dlc_label_for_aep
import os
import copy
import weis.inputs as sch
import numpy as np

class TestIECWind(unittest.TestCase):

    def test_aep_fallback_label_supports_dlc_16(self):
        DLCs = ['1.6']
        np.testing.assert_equal(get_dlc_label_for_aep(DLCs), '1.6')

    def test_n_ws_aep_counts_dlc_16(self):

        ws_cut_in = 4.
        ws_cut_out = 25.
        ws_rated = 10.
        wind_speed_class = 'I'
        wind_turbulence_class = 'B'

        weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + os.sep
        fname_modeling_options = os.path.join(weis_dir, 'examples', '05_IEA-3.4-130-RWT', 'modeling_options.yaml')
        modeling_options = sch.load_modeling_yaml(fname_modeling_options)

        dlc_opt = copy.deepcopy(modeling_options['DLC_driver']['DLCs'][0])
        dlc_opt['DLC'] = '1.6'

        dlc_generator = DLCGenerator(
            ws_cut_in,
            ws_cut_out,
            ws_rated,
            wind_speed_class,
            wind_turbulence_class,
            modeling_options['DLC_driver']['fix_wind_seeds'],
            modeling_options['DLC_driver']['fix_wave_seeds'],
            modeling_options['DLC_driver']['metocean_conditions'],
            modeling_options['DLC_driver']
        )
        dlc_generator.generate(dlc_opt['DLC'], dlc_opt)

        dlc_label_for_aep = get_dlc_label_for_aep(['1.6'])
        dlc_aep_ws = [c.URef for c in dlc_generator.cases if c.label == dlc_label_for_aep]
        n_ws_aep = len(np.unique(dlc_aep_ws))

        np.testing.assert_equal(n_ws_aep, 6)
        np.testing.assert_allclose(
            dlc_generator.qblade_case_inputs[-1][('QSim', 'HORANGLE')]['vals'],
            np.zeros(dlc_generator.n_cases),
        )

    def test_custom_dlc_probabilities_follow_seed_expansion(self):

        ws_cut_in = 4.
        ws_cut_out = 25.
        ws_rated = 10.
        wind_speed_class = 'I'
        wind_turbulence_class = 'B'

        weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + os.sep
        fname_modeling_options = os.path.join(weis_dir, 'examples', '05_IEA-3.4-130-RWT', 'modeling_options.yaml')
        modeling_options = sch.load_modeling_yaml(fname_modeling_options)

        dlc_opt = copy.deepcopy(modeling_options['DLC_driver']['DLCs'][0])
        dlc_opt.update({
            'DLC': 'Custom',
            'n_seeds': 2,
            'wind_speed': [6.0, 8.0],
            'wave_height': [1.1, 2.2],
            'wave_period': [6.8, 8.8],
            'wave_heading': [0.0, 30.0],
            'HORANGLE': [5.0, 10.0],
            'probabilities': [0.2, 0.8],
            'yaw_misalign': [0.0],
        })

        dlc_generator = DLCGenerator(
            ws_cut_in,
            ws_cut_out,
            ws_rated,
            wind_speed_class,
            wind_turbulence_class,
            modeling_options['DLC_driver']['fix_wind_seeds'],
            modeling_options['DLC_driver']['fix_wave_seeds'],
            modeling_options['DLC_driver']['metocean_conditions'],
            modeling_options['DLC_driver']
        )
        dlc_generator.generate(dlc_opt['DLC'], dlc_opt)

        np.testing.assert_equal(dlc_generator.n_cases, 4)
        np.testing.assert_allclose([c.URef for c in dlc_generator.cases], [6.0, 6.0, 8.0, 8.0])
        np.testing.assert_allclose([c.wave_height for c in dlc_generator.cases], [1.1, 1.1, 2.2, 2.2])
        np.testing.assert_allclose([c.wave_period for c in dlc_generator.cases], [6.8, 6.8, 8.8, 8.8])
        np.testing.assert_allclose([c.wave_heading for c in dlc_generator.cases], [0.0, 0.0, 30.0, 30.0])
        np.testing.assert_allclose([c.probability for c in dlc_generator.cases], [0.1, 0.1, 0.4, 0.4])
        np.testing.assert_allclose(sum(c.probability for c in dlc_generator.cases), 1.0)
        np.testing.assert_allclose(
            dlc_generator.qblade_case_inputs[-1][('QSim', 'HORANGLE')]['vals'],
            [5.0, 5.0, 10.0, 10.0],
        )

    def test_generator(self):

        # Wind turbine inputs that will eventually come in from somewhere
        ws_cut_in = 4.
        ws_cut_out = 25.
        ws_rated = 10.
        wind_speed_class = 'I'
        wind_turbulence_class = 'B'

        # Load modeling options file
        weis_dir                = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep
        fname_modeling_options = os.path.join(weis_dir , "examples", "05_IEA-3.4-130-RWT", "modeling_options.yaml")
        modeling_options = sch.load_modeling_yaml(fname_modeling_options)
        
        # Extract user defined list of cases
        DLCs = modeling_options['DLC_driver']['DLCs']
        
        # Initialize the generator
        fix_wind_seeds = modeling_options['DLC_driver']['fix_wind_seeds']
        fix_wave_seeds = modeling_options['DLC_driver']['fix_wave_seeds']
        metocean = modeling_options['DLC_driver']['metocean_conditions']
        dlc_generator = DLCGenerator(
            ws_cut_in, 
            ws_cut_out, 
            ws_rated, 
            wind_speed_class, 
            wind_turbulence_class, 
            fix_wind_seeds, 
            fix_wave_seeds, 
            metocean,
            modeling_options['DLC_driver']
            )

        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)
            

        np.testing.assert_equal(dlc_generator.cases[11].URef, ws_cut_out)
        np.testing.assert_equal(dlc_generator.n_cases, 60)

        # Determine wind speeds that will be used to calculate AEP (using DLC AEP or 1.1)
        DLCs = [i_dlc['DLC'] for i_dlc in modeling_options['DLC_driver']['DLCs']]
        if 'AEP' in DLCs:
            DLC_label_for_AEP = 'AEP'
        else:
            DLC_label_for_AEP = '1.1'
        dlc_aep_ws = [c.URef for c in dlc_generator.cases if c.label == DLC_label_for_AEP]
        n_ws_aep = len(np.unique(dlc_aep_ws))

        np.testing.assert_equal(n_ws_aep, 6)
        
if __name__ == "__main__":
    unittest.main()
