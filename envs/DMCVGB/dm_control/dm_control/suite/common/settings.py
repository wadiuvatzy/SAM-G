import os
import numpy as np
from dm_control.suite import common
from dm_control.utils import io as resources
import xmltodict

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "./common/materials.xml",
    "./common/skybox.xml",
    "./common/visual.xml",
]


def get_model_and_assets_from_setting_kwargs(model_fname, setting_kwargs=None):
    """"Returns a tuple containing the model XML string and a dict of assets."""
    assets = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}

    if setting_kwargs is None:
        return common.read_model(model_fname), assets

    # Convert XML to dicts
    model = xmltodict.parse(common.read_model(model_fname))
    materials = xmltodict.parse(assets['./common/materials.xml'])
    skybox = xmltodict.parse(assets['./common/skybox.xml'])

    # Edit grid floor
    if 'grid_rgb1' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_rgb1'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["grid_rgb1"][0]} {setting_kwargs["grid_rgb1"][1]} {setting_kwargs["grid_rgb1"][2]}'
    if 'grid_rgb2' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_rgb2'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["grid_rgb2"][0]} {setting_kwargs["grid_rgb2"][1]} {setting_kwargs["grid_rgb2"][2]}'
    if 'grid_markrgb' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_markrgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["grid_markrgb"][0]} {setting_kwargs["grid_markrgb"][1]} {setting_kwargs["grid_markrgb"][2]}'
    if 'grid_texrepeat' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_texrepeat'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['material'][0]['@texrepeat'] = \
            f'{setting_kwargs["grid_texrepeat"][0]} {setting_kwargs["grid_texrepeat"][1]}'

    # Edit self
    if 'self_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['self_rgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['material'][1]['@name'] == 'self'
        materials['mujoco']['asset']['material'][1]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0]} {setting_kwargs["self_rgb"][1]} {setting_kwargs["self_rgb"][2]} 1'
    # if 'self_robo_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['self_rgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['material'][13]['@name'] == 'white'
        assert materials['mujoco']['asset']['material'][14]['@name'] == 'dark'
        materials['mujoco']['asset']['material'][13]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0]-0.7+1} {setting_kwargs["self_rgb"][1]-0.5+1} {setting_kwargs["self_rgb"][2]-0.3+1} 1'
        materials['mujoco']['asset']['material'][14]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0] - 0.7 + 0.2} {setting_kwargs["self_rgb"][1] - 0.5 + 0.2} {setting_kwargs["self_rgb"][2] - 0.3 + 0.2} 1'
        materials['mujoco']['asset']['material'][15]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0] - 0.7 + 0.25} {setting_kwargs["self_rgb"][1] - 0.5 + 0.25} {setting_kwargs["self_rgb"][2] - 0.3 + 0.25} 1'

    if setting_kwargs.__contains__('self_rgb1') and setting_kwargs.__contains__('self_rgb'):
        # assert model['mujoco']['asset']['material'][1]['@name'] == 'table_mat'
        materials['mujoco']['asset']['material'][16]['@rgba'] = \
            f'{1-setting_kwargs["self_rgb"][0]} {1-setting_kwargs["self_rgb"][1]} {1-setting_kwargs["self_rgb"][2]} 1'
    if setting_kwargs.__contains__('table_texture'):
        assert model['mujoco']['asset']['material'][1]['@name'] == 'table_mat'
        model['mujoco']['asset']['material'][1]['@texture'] = setting_kwargs['table_texture']

    if setting_kwargs.__contains__('ground_texture'):
        assert model['mujoco']['asset']['material']['@name'] == 'groundplane'
        model['mujoco']['asset']['material']['@texture'] = setting_kwargs['ground_texture']
    # Edit skybox
    if 'skybox_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_rgb'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["skybox_rgb"][0]} {setting_kwargs["skybox_rgb"][1]} {setting_kwargs["skybox_rgb"][2]}'
    if 'skybox_rgb2' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_rgb2'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["skybox_rgb2"][0]} {setting_kwargs["skybox_rgb2"][1]} {setting_kwargs["skybox_rgb2"][2]}'
    if 'skybox_markrgb' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_markrgb'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["skybox_markrgb"][0]} {setting_kwargs["skybox_markrgb"][1]} {setting_kwargs["skybox_markrgb"][2]}'

    # Convert back to XML
    model_xml = xmltodict.unparse(model)
    assets['./common/materials.xml'] = xmltodict.unparse(materials)
    assets['./common/skybox.xml'] = xmltodict.unparse(skybox)

    return model_xml, assets
