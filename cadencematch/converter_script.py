from aubio import source, tempo
from pathlib import Path
from math import fabs
from numpy import diff, median
from soundfile import read, write
import pyrubberband
import glob
import os

def get_file_bpm(path):
    windowsize, hopsize = 1024, 512

    original = source(path)
    samplerate = original.samplerate
    o = tempo("default", windowsize, hopsize, samplerate)
    beats = []
    total_frames = 0

    while True:
        samples, read = original()
        is_beat = o(samples)
        if is_beat:
            this_beat = o.get_last_s()
            beats.append(this_beat)
        total_frames += read
        if read < hopsize:
            break

    if len(beats) > 1:
        bpms = 60./diff(beats)
        return median(bpms),samplerate
    else:
        return 0,samplerate

def closest_bpm_match(bpm,targetbpm,rel):
    singletempo = (targetbpm-bpm)/bpm
    doubletempo = (2*targetbpm-bpm)/bpm
    if (min(fabs(singletempo),fabs(doubletempo)) < rel):
        if (fabs(singletempo) < fabs(doubletempo)):
            return singletempo
        else:
            return doubletempo
    else:
        return 0

musicfiles = glob.glob("input/*.wav")
musicfiles = sorted(musicfiles)
total = len(musicfiles)
print("Input folder contains %d .wav files." % (total))
targetbpm = float(input("Specify desired BPM value:"))
rel = input("Specify maximum relative change of tempo (0-1, default is 0.35):")
try:
    rel = float(rel)
except ValueError:
    print("Falling back to default value.")
    rel = 0.35

for i,track in enumerate(musicfiles):
    bpm,samplerate = get_file_bpm(track)
    if (bpm == 0):
        print("Unable to determine track tempo. Skipping.")
        continue
    speedupfactor = closest_bpm_match(bpm,targetbpm,rel)
    if (speedupfactor == 0):
        print("Track %d/%d cannot be converted without severe distortion. Skipping." % (i+1,total)))
        continue
    finalspeed = 1.0 + speedupfactor

    song, samplerate = read(track)
    stretched = pyrubberband.time_stretch(song, samplerate, finalspeed)
    pitch_corrected = pyrubberband.pitch_shift(stretched, samplerate, finalspeed)
    outpath = "output/" + "%dBPM_" % (int(targetbpm)) + Path(track).stem + ".wav"
    write(outpath,pitch_corrected, samplerate, format="wav")
    print("Successfully converted track %d/%d." %(i+1,total))