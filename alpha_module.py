import os, typing, json
from enum import Enum
from pandas import DataFrame

DATABASE = "/Users/austin/Documents/WQ_Research_Infra"
PNL_FOLDER = os.path.join(DATABASE, 'pnl')
TVR_FOLDER = os.path.join(DATABASE, 'tvr')

class AlphaStage(Enum):
    PENDING: str = os.path.join(DATABASE, 'pending')
    RUNNING: str = os.path.join(DATABASE, 'running')
    COMPLETED: str = os.path.join(DATABASE, 'complete')

class Alpha:
    def __init__(self, name: str, payload, alpha_stage: AlphaStage = AlphaStage.PENDING, running_data: typing.Dict[str, str] = None, response_data: typing.Dict = None, load_from_file: bool = False) -> None:
        self.name = name
        self.payload = payload
        self.alpha_stage = alpha_stage
        self.running_data = running_data
        self.response_data = response_data

        self.file_path = os.path.join(AlphaStage.PENDING.value, self.name)
        
    @classmethod
    def load_from_disk(cls, file_path:str):
        with open(file_path, "r") as handle:
            alpha_dict = json.load(handle)
        return cls(name=alpha_dict['name'], payload=alpha_dict['payload'], alpha_stage=AlphaStage(alpha_dict["alpha_stage"]), running_data=alpha_dict["running_data"], response_data=alpha_dict["response_data"], load_from_file=True)

    def _to_dict(self) -> typing.Dict:
        return {
            'name': self.name,
            'payload': self.payload,
            'alpha_stage': self.alpha_stage.value,
            'running_data': self.running_data,
            'response_data': self.response_data
        }

    def save_to_disk(self):
        self.file_path = os.path.join(self.alpha_stage.value, self.name)
        with open(self.file_path, "w") as handle:
            json.dump(self._to_dict(), handle)

    def update_pnl_path(self, pnl_path):
        self.response_data['pnl_path'] = pnl_path

    def update_tvr_path(self, tvr_path):
        self.response_data['tvr_path'] = tvr_path

    def update_stage(self, alpha_stage: AlphaStage):
        self.alpha_stage = alpha_stage

    def update_location_url(self, location_url: str):
        self.running_data['location_url'] = location_url

    def update_running_data(self, running_data):
        self.running_data = running_data
    def update_response_data(self, response_data):
        self.response_data = response_data