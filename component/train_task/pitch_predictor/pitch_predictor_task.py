from component.train_task.pitch_predictor.dataset import PitchPredictorDataset
from tasks.tts.fs2 import FastSpeech2Task


class PitchPredictorTask(FastSpeech2Task):
    def __init__(self):
        super().__init__()
        self.dataset_cls = PitchPredictorDataset