import torch
import os
import json
import numpy as np
import csv
import pretty_midi as pm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## some initialize ##
[DRUMMER, SESSION, ID, STYLE, BPM, BEAT_TYPE, TIME_SIGNATURE, MIDI_FILENAME, AUDIO_FILENAME, DURATION, SPLIT] = [0,1,2,3,4,5,6,7,8,9,10]
with open('model_config.json') as f:
    args = json.load(f)

class FindMIDI():
    """ Find MIDI file path and its BPM(int)"""
    def __init__(self):
        """Initialize FindMIDI."""
        pass

    def read_csv(self, path):
        info = []
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                info.append(row)
        info = info[1:-1]
        info = [a for a in info if a != []]
        info_T = np.array(info).T.tolist()

        return info_T

    def get_path_and_BPM(self, path):
        data_list = []
        midi_filenames = []
        BPM_list = []
        data_list = self.read_csv(path)

        for index in range(len(data_list[MIDI_FILENAME])):
            midi_filenames.append("../../groove/onesec_midi/" + data_list[MIDI_FILENAME][index])
            BPM_list.append(int(data_list[BPM][index]))

        return midi_filenames, BPM_list

    def __call__(self, path):
        """
        Args:
            path (str) : csv file containing MIDI file information.

        """
        return self.get_path_and_BPM(path)

class LoadMIDI():
    """ load MIDI file as PrettyMIDI object from midi_filename """
    def __init__(self):
        """ Initialize LoadMIDI """
        pass

    def get_MIDI_object(self, path):
        midi_pattern = pm.PrettyMIDI(path)
        instruments = midi_pattern.instruments

        return instruments

    def __call__(self, path):
        """
        Args:
            path (str): MIDI filepath.
        """
        midi = self.get_MIDI_object(path)
        return midi


class make_feature_ndarray():
    def __init__(self):
        pass

    def make_feature(self, inst, steps, BPM):
        MultiHot = np.zeros((steps, 81))
        MultiHot = MultiHot.tolist()
        notes = inst[0].notes
        section = 2 * (60 * 4) / BPM  # 2小節 = 1 section
        beat = section / steps
        for note in notes:
            for step in range(steps):
                if note.start - beat * step < 0:
                    position = note.start - (step - 1) * beat
                    if position >= 0 and position <= beat/2:
                        MultiHot[step-1][note.pitch] = note.velocity/127
                        break
                    elif position >= beat/2 and position <= beat:
                        MultiHot[step][note.pitch] = note.velocity/127
                        break

        return np.array(MultiHot)

    def __call__(self, inst, steps, BPM):
        """
        Args:
            inst (PrettyMIDI.instrument object) : MIDI Instrument object
            steps (int) : Time resolution of 1 section
            BPM (int) : BPM of MIDI instrument
        """
        features = self.make_feature(inst, steps, BPM)
        return features

class MusicArrayDataset(Dataset):
    """Music array dataset."""

    def __init__(self, root_path, time_step):
        """Initialize MusicArrayDataset.

        Args:
            root_path (str): Root csv information path of midi files.
            time_step (int) : Time resolution of 1 section.
            transform (function) : Transform dataset.

        """
        self.time_step = time_step

        self.load_midi_file = LoadMIDI()
        self.make_feature_ndarray = make_feature_ndarray()

        # MIDIファイルのパスと対応するMIDIファイルのBPMを取得して格納する
        find_filenames_and_BPM_from_csv = FindMIDI()
        self.midi_filenames , self.BPM_list = find_filenames_and_BPM_from_csv(root_path)


    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            ndarray: MultiHotVector (per time resolution) list of 1 section (T, D)
                where T is 64 (time resolution). and D is 81 (kinds of instruments).

        """
        # MIDI ファイルを読み込む
        midi_filename = self.midi_filenames[idx]
        BPM = self.BPM_list[idx]
        midi = self.load_midi_file(midi_filename)

        # 読み込んだMIDIファイルを加工してndarrayにする
        steps = self.time_step
        features = self.make_feature_ndarray(midi, steps, BPM)

        return features


    def __len__(self):
        """Return dataset length.

        Returns:
            int: Length of dataset.

        """
        return len(self.midi_filenames)


class Collater(object):
    """Collater to convert the list of batch into tensors."""

    def __init__(self, chunk_size):
        """Initialize collater.

        Args:
            chunk_size (int): Target ength of each item in batch (たぶん?).

        """
        self.chunk_size = chunk_size

    def __call__(self, batch):
        # batch は自分で定義したDatasetの__getitem__の返り値のlist
        # 例えば長さの異なる特徴量のリストになる
        # [(T_1, D), (T_2, D), ..., (T_batchsize, D)]
        new_batch = []
        for b in batch:
            # 各アイテムは固定長(64,81)
            # ランダムに切り出す部分だけ実装
            if b.shape[0] < self.chunk_size:
                print("Shorter than chunk size. Skipped.")
                continue
            # ランダムに開始位置を決定して切り出す
            start_offset = np.random.randint(0, b.shape[0] - self.chunk_size + 1)
            b_ = b[start_offset: start_offset + self.chunk_size]

            new_batch.append(b_)

        # Tensorに変換
        new_batch = torch.FloatTensor(new_batch)

        return new_batch

chunk_size = args['time_step']

class MusicArrayLoader():
    def __init__(self, data_path, time_step, chunk_size):
        self.path = data_path
        self.step = time_step
        self.chunk_size = chunk_size

    def data_loader(self, batch_size):
        my_dataset = MusicArrayDataset(self.path, self.step)
        my_collater_fn = Collater(self.chunk_size)
        MusicArrayLoader = DataLoader(
            dataset = my_dataset,
            collate_fn = my_collater_fn,
            shuffle = True,
            batch_size = batch_size,
            )
        return MusicArrayLoader

    def get_batch(self, batch_size):
        for batch in self.data_loader(batch_size):
            break
        return batch

"""
# check dataloader output
for batch in MusicArrayLoader:
    print(batch)
    
    ################################################
    ## check array                                ##
    ## batch_ar = batch.numpy()                   ##
    ## np.set_printoptions(threshold=np.inf)      ##
    ## print(batch_ar)                            ##
    ################################################

    break

"""