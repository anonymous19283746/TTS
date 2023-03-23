from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=Path,
        nargs='+',
        default=[
            Path('../data/saspeech_gold_standard_resampled/wavs/'),
            Path('../data/saspeech_automatic_data_resampled/wavs/')
        ]
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='../data/saspeech_windowed/'
    )

    parser.add_argument(
        '--window_max_length',
        type=float,
        default=5,
        help='max length of a window of audio, in seconds'
    )

    parser.add_argument(
        '--overlap_size',
        type=float,
        default=1,
        help='length of the overlap between adjacent windows, in seconds'
    )

    parser.add_argument(
        '--min_window_count',
        type=int,
        default=3,
        help='number of windows that must fit in a .wav in order to be split'
    )

    args = parser.parse_args()

    source_files = []
    for source_file_dir in args.input_dir:
        source_files.extend(sorted(source_file_dir.glob('*wav')))

    output_path = args.output_dir
    output_path.mkdir(exist_ok=True)

    window_max_length = args.window_max_length
    overlap_size = args.overlap_size
    min_window_count = args.min_window_count

    window_length_no_overlap = window_max_length - overlap_size

    for wav_path in tqdm(source_files):
        data, sr = sf.read(wav_path)
        # length in seconds of the file
        audio_length = len(data) / sr

        window_count = int(np.ceil(audio_length / window_length_no_overlap))

        # if file too short to split, simply copy it
        if window_count < min_window_count:
            out_file_path = output_path / wav_path.name
            #print(f"file {wav_path.name} too short to split ({audio_length} secs), copying as is!")
            shutil.copy2(wav_path, out_file_path)
            continue

        # half of the overlap size, but in samples, not seconds
        half_overlap_samples = int(overlap_size * sr / 2)

        # We split the file into window_count windows of roughly equal size.
        # this is because we don't have a remainder this way, unlike fixed-size
        # windows
        
        # we have window_count + 1 borders in our file including its start and end
        borders = np.linspace(0, len(data), window_count + 1).astype(int)

        # first window starts at 0
        # all the other windows start at borders - half overlap size
        starts = np.append(0, borders[1:-1] - half_overlap_samples)
        # last window ends at end of file
        # the rest of the windows end at borders + half overlap size
        ends = np.append(borders[1:-1] + half_overlap_samples, borders[-1])
        for i, (start, end) in enumerate(zip(starts, ends)):
            out_file_path = output_path / f"{wav_path.stem}_{i:03}.wav"
            sf.write(out_file_path, data[start:end], sr)

if __name__=='__main__':
    main()