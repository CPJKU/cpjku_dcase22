import librosa
import numpy as np
from ba3l.ingredients.ingredient import Ingredient


default_processor = Ingredient('default_processor')

@default_processor.command
def get_processor_default(n_fft=2048 * 2, sr=44100, log_spec=False, n_mels=256, win_length=None, hop_length=512 * 2,
                          resample_only=False):
    """
    @param n_fft: number of FFT applied to the window
    @param sr: sampling rate of the audio
    @param log_spec: boolean value to indiciate log_spectrograms
    @param n_mels: number of mels for the preprocessing
    @param win_length: window size of the fft
    @param hop_length: hop size of the fft
    @param resample_only: boolean value indicating that preprocess only does the resampling or creates spectograms
    @return: spectorgrams or resampled version of the audio
    """
    if resample_only:
        print("get_processor_default: resample_only")
        print(sr)
    else:
        print("get_processor_default:", n_fft, sr, n_mels, hop_length)

    def do_process(file_path):
        # this is the slowest part resampling
        sig, _ = librosa.load(file_path, sr=sr, mono=True)
        sig = sig[np.newaxis]
        if resample_only:
            return sig

        spectrograms = []
        for y in sig:

            # compute stft
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
                                center=True, pad_mode='reflect')

            # keep only amplitudes
            stft = np.abs(stft)

            # spectrogram weighting
            if log_spec:
                stft = np.log10(stft + 1)
            else:
                freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
                stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)

            # apply mel filterbank
            spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=None)

            # keep spectrogram
            spectrograms.append(np.asarray(spectrogram))

        spectrograms = np.asarray(spectrograms, dtype=np.float32)
        return spectrograms

    return do_process


# indentifier for the type of dataset.
@default_processor.config
def default_processor_def_config(n_fft, sr, log_spec, n_mels, hop_length, resample_only):
    if resample_only:
        identifier = f"resample{sr}"
    else:
        identifier = f"mel{n_mels}_f{n_fft}_sr{sr}_h{hop_length}" + ("_log" if log_spec else "")
