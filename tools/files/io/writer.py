import os
import shutil
import logging
from typing import Any
import pandas as pd
from pathlib import Path

from .format import Format, SUFFIX

_logger = logging.getLogger(__name__)


class Writer:
    """ The study folder reader class """

    _parsers: dict = {}

    @classmethod
    def register_parser(cls, format, parser):
        """Registers external parser"""

        if format in cls._parsers:
            _logger.warning(f'Parser {format} already exists. Gets replaced.')
        cls._parsers[format] = parser


    def write(self, path:Path, data:Any):

        path = Path(path)

        if path.is_dir():
            return self.read_dir(path, format)
        else:
            return self.read_file(path, format)
    # Private

    def _get_path(self, node: Format, *args) -> str:
        """Gets the node filepath"""

        file_pattern = NODE_PATH[node].format(*args)
        return os.path.join(self._path, file_pattern)

    def _set_node(self, node: Format, data: Any, *args):
        """Reads the node object"""

        path = self._get_path(node, *args)
        parser = self._parsers[node]
        if not parser:
            _logger.error(f'Parser for node {node} is not registered.')
            return None

        try:
            return parser(path, data)
        except Exception as e:
            _logger.error(f'Parser for node {node} write exception: {e}.')
            return None

    # Public

    @property
    def path(self):
        return self._path

    # Dirs

    def get_temp(self, respondent_uid: str = None) -> Path:
        return self._get_path(
            Format.TEMP) if respondent_uid is None else self._get_path(
                Format.TEMP_RESPONDENT, respondent_uid)

    def mk_temp(self, respondent_uid: str = None):
        path = self.get_temp(respondent_uid)
        Path(path).mkdir(parents=True, exist_ok=True)

    def rm_temp(self, respondent_uid: str = None):
        try:
            path = self._get_path(
                Format.TEMP) if respondent_uid is None else self._get_path(
                    Format.TEMP_RESPONDENT, respondent_uid)
            shutil.rmtree(path)
        except Exception as e:
            _logger.error(f'Removing temp folder exception: {e}.')


    def gaze_raw_path(self, respondent_uid: str, stimuli_name: str):
        return self._get_path(Format.GAZE_RAW, respondent_uid, stimuli_name)

    def respiration_raw_path(self, respondent_uid: str, stimuli_name: str):
        return self._get_path(Format.RESPIRATION_RAW, respondent_uid, stimuli_name)

    def face_raw_path(self, respondent_uid: str, stimuli_name: str):
        return self._get_path(Format.FACE_RAW, respondent_uid, stimuli_name)

    def pose_raw_path(self, respondent_uid: str, stimuli_name: str):
        return self._get_path(Format.POSE_RAW, respondent_uid, stimuli_name)

    # Data

    def set_gaze_raw(self, respondent_uid: str, stimuli_name: str,
                 data: pd.DataFrame):
        return self._set_node(Format.GAZE_RAW, data, respondent_uid, stimuli_name)

    def set_gaze_map(self, respondent_uid: str, stimuli_name: str,
                 data: pd.DataFrame):
        return self._set_node(Format.GAZE_MAP, data, respondent_uid, stimuli_name)

    def set_respiration_raw(self, respondent_uid: str, stimuli_name: str,
                 data: pd.DataFrame):
        return self._set_node(Format.RESPIRATION_RAW, data, respondent_uid, stimuli_name)

    def set_calibration(self, respondent_uid: str, data: pd.DataFrame):
        return self._set_node(Format.CALIBRATION, data, respondent_uid)

    def set_et_eyetracker(self, respondent_uid: str, data: pd.DataFrame):
        return self._set_node(Format.ET_EYETRACKER, data, respondent_uid)

    def set_et_metrics(self, respondent_uid: str, data: pd.DataFrame):
        return self._set_node(Format.ET_METRICS, data, respondent_uid)

    def set_et_respiration(self, respondent_uid: str, data: pd.DataFrame):
        return self._set_node(Format.ET_RESPIRATION, data, respondent_uid)

    def set_face_raw(self, respondent_uid: str, stimuli_name: str, data):
        return self._set_node(Format.FACE_RAW, data, respondent_uid, stimuli_name)

    def set_face(self, respondent_uid: str, stimuli_name: str, path: str):
        return self._set_node(Format.FACE, path, respondent_uid, stimuli_name)

    def set_pose_raw(self, respondent_uid: str, stimuli_name: str, data):
        return self._set_node(Format.POSE_RAW, data, respondent_uid, stimuli_name)

    def set_pose(self, respondent_uid: str, stimuli_name: str, path: str):
        return self._set_node(Format.POSE, path, respondent_uid, stimuli_name)

