import json
import os
import fastapi
import uvicorn

from utils.hparams_v2 import set_hparams
from . import config

class WebHandler:
    def __init__(self, exp_name):
        # model
        self.exp_name = exp_name
        self.hparams = set_hparams(
            exp_name=exp_name,
            task="svs",
            make_work_dir=False
        )
        self.work_dir = self.hparams["work_dir"]
        self.spk_map = self.build_spk_map()
        self.lang_map = self.build_lang_map()
        # web
        self.app = fastapi.FastAPI()
        self.app.add_api_route(config.get_languages_api, self.api_get_languages, methods=['GET'])
        self.app.add_api_route(config.get_speakers_api, self.api_get_speakers, methods=['GET'])
        

    def build_spk_map(self):
        spk_map_fn = os.path.join(self.work_dir, 'spk_map.json')
        assert os.path.exists(spk_map_fn), f"Speaker map file {spk_map_fn} not found"
        with open(spk_map_fn, 'r') as f:
            spk_map = json.load(f)
        return spk_map
        
    def build_lang_map(self):
        lang_map_fn = os.path.join(self.work_dir, 'lang_map.json')
        assert os.path.exists(lang_map_fn), f"Language map file {lang_map_fn} not found"
        with open(lang_map_fn, 'r') as f:
            lang_map = json.load(f)
        return lang_map

    async def api_get_languages(self):
        return list(self.lang_map.keys())
    
    async def api_get_speakers(self):
        return list(self.spk_map.keys())

    def handle(self):
        uvicorn.run(self.app, host=config.server_host, port=config.server_port)